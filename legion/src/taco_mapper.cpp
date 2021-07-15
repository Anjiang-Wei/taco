#include "taco_mapper.h"
#include "mappers/logging_wrapper.h"

using namespace Legion;
using namespace Legion::Mapping;

const char* TACOMapperName = "TACOMapper";

void register_taco_mapper(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs) {
  for (auto it : local_procs) {
#ifdef TACO_USE_LOGGING_MAPPER
    runtime->replace_default_mapper(new Mapping::LoggingWrapper(new TACOMapper(runtime->get_mapper_runtime(), machine, it, TACOMapperName)), it);
#else
    runtime->replace_default_mapper(new TACOMapper(runtime->get_mapper_runtime(), machine, it, TACOMapperName), it);
#endif
  }
}

TACOMapper::TACOMapper(Legion::Mapping::MapperRuntime *rt, Legion::Machine &machine, const Legion::Processor &local, const char* name)
    : DefaultMapper(rt, machine, local, name) {
  {
    int argc = Legion::HighLevelRuntime::get_input_args().argc;
    char **argv = Legion::HighLevelRuntime::get_input_args().argv;
    for (int i = 1; i < argc; i++) {
#define BOOL_ARG(argname, varname) do {       \
          if (!strcmp(argv[i], (argname))) {    \
            varname = true;                   \
            continue;                         \
          } } while(0);
      BOOL_ARG("-tm:fill_cpu", this->preferCPUFill);
      BOOL_ARG("-tm:validate_cpu", this->preferCPUValidate);
      BOOL_ARG("-tm:untrack_valid_regions", this->untrackValidRegions);
      BOOL_ARG("-tm:numa_aware_alloc", this->numaAwareAllocs);
#undef BOOL_ARG
    }
  }

  // Record for each OpenMP processor what NUMA region is the closest.
  for (auto proc : this->local_omps) {
    Machine::MemoryQuery local(this->machine);
    local.local_address_space()
         .only_kind(Memory::SOCKET_MEM)
         .best_affinity_to(proc)
         ;
    if (local.count() > 0) {
      this->numaDomains[proc] = local.first();
    }
  }
}

void TACOMapper::select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                         const Legion::Task &task,
                                         const SelectShardingFunctorInput &input,
                                         SelectShardingFunctorOutput &output) {
  // See if there is something special that we need to do. Otherwise, return
  // the TACO sharding functor.
  if ((task.tag & PLACEMENT_SHARD) != 0) {
    int *args = (int *) (task.args);
    // TODO (rohany): This logic makes it look like an argument
    //  serializer / deserializer like is done in Legate would be helpful.
    // The shard ID is the first argument. The generated code registers the desired
    // sharding functor before launching the task.
    Legion::ShardingID shardingID = args[0];
    output.chosen_functor = shardingID;
  } else {
    output.chosen_functor = TACOShardingFunctorID;
  }
}

bool TACOMapper::tm_create_custom_instances(Legion::Mapping::MapperContext ctx,
                                            Legion::Processor target, Legion::Memory target_memory,
                                            const Legion::RegionRequirement &req, unsigned index,
                                            std::set<Legion::FieldID> &needed_fields, // will destroy
                                            const Legion::TaskLayoutConstraintSet &layout_constraints,
                                            Legion::LayoutConstraintSet constraints,
                                            bool needs_field_constraint_check,
                                            std::vector<Legion::Mapping::PhysicalInstance> &instances,
                                            size_t *footprint) {
  // Special case for reduction instances, no point in checking
  // for existing ones and we also know that currently we can only
  // make a single instance for each field of a reduction
  if (req.privilege == LEGION_REDUCE)
  {
    // Iterate over the fields one by one for now, once Realm figures
    // out how to deal with reduction instances that contain
    bool force_new_instances = true; // always have to force new instances
    LayoutConstraintID our_layout_id =
        default_policy_select_layout_constraints(ctx, target_memory, req,
                                                 TASK_MAPPING, needs_field_constraint_check, force_new_instances);
    LayoutConstraintSet our_constraints =
        runtime->find_layout_constraints(ctx, our_layout_id);
    instances.resize(instances.size() + req.privilege_fields.size());
    unsigned idx = 0;
    for (std::set<FieldID>::const_iterator it =
        req.privilege_fields.begin(); it !=
                                      req.privilege_fields.end(); it++, idx++)
    {
      our_constraints.field_constraint.field_set.clear();
      our_constraints.field_constraint.field_set.push_back(*it);
      if (!default_make_instance(ctx, target_memory, our_constraints,
                                 instances[idx], TASK_MAPPING, force_new_instances,
                                 true/*meets*/, req, footprint))
        return false;
    }
    return true;
  }
  // Before we do anything else figure out our
  // constraints for any instances of this task, then we'll
  // see if these constraints conflict with or are satisfied by
  // any of the other constraints
  bool force_new_instances = false;
  LayoutConstraintID our_layout_id =
      default_policy_select_layout_constraints(ctx, target_memory, req,
                                               TASK_MAPPING, needs_field_constraint_check, force_new_instances);
  const LayoutConstraintSet &our_constraints =
      runtime->find_layout_constraints(ctx, our_layout_id);
  for (std::multimap<unsigned,LayoutConstraintID>::const_iterator lay_it =
      layout_constraints.layouts.lower_bound(index); lay_it !=
                                                     layout_constraints.layouts.upper_bound(index); lay_it++)
  {
    // Get the constraints
    const LayoutConstraintSet &index_constraints =
        runtime->find_layout_constraints(ctx, lay_it->second);
    std::vector<FieldID> overlapping_fields;
    const std::vector<FieldID> &constraint_fields =
        index_constraints.field_constraint.get_field_set();
    if (!constraint_fields.empty())
    {
      for (unsigned idx = 0; idx < constraint_fields.size(); idx++)
      {
        FieldID fid = constraint_fields[idx];
        std::set<FieldID>::iterator finder = needed_fields.find(fid);
        if (finder != needed_fields.end())
        {
          overlapping_fields.push_back(fid);
          // Remove from the needed fields since we're going to handle it
          needed_fields.erase(finder);
        }
      }
      // If we don't have any overlapping fields, then keep going
      if (overlapping_fields.empty())
        continue;
    }
    else // otherwise it applies to all the fields
    {
      overlapping_fields.insert(overlapping_fields.end(),
                                needed_fields.begin(), needed_fields.end());
      needed_fields.clear();
    }
    // Now figure out how to make an instance
    instances.resize(instances.size()+1);
    // Check to see if these constraints conflict with our constraints
    // or whether they entail our mapper preferred constraints
    if (runtime->do_constraints_conflict(ctx, our_layout_id, lay_it->second)
        || runtime->do_constraints_entail(ctx, lay_it->second, our_layout_id))
    {
      // They conflict or they entail our constraints so we're just going
      // to make an instance using these constraints
      // Check to see if they have fields and if not constraints with fields
      if (constraint_fields.empty())
      {
        LayoutConstraintSet creation_constraints = index_constraints;
        creation_constraints.add_constraint(
            FieldConstraint(overlapping_fields,
                            index_constraints.field_constraint.contiguous,
                            index_constraints.field_constraint.inorder));
        if (!default_make_instance(ctx, target_memory, creation_constraints,
                                   instances.back(), TASK_MAPPING, force_new_instances,
                                   true/*meets*/, req, footprint))
          return false;
      }
      else if (!default_make_instance(ctx, target_memory, index_constraints,
                                      instances.back(), TASK_MAPPING, force_new_instances,
                                      false/*meets*/, req, footprint))
        return false;
    }
    else
    {
      // These constraints don't do as much as we want but don't
      // conflict so make an instance with them and our constraints
      LayoutConstraintSet creation_constraints = index_constraints;
      default_policy_select_constraints(ctx, creation_constraints,
                                        target_memory, req);
      creation_constraints.add_constraint(
          FieldConstraint(overlapping_fields,
                          creation_constraints.field_constraint.contiguous ||
                          index_constraints.field_constraint.contiguous,
                          creation_constraints.field_constraint.inorder ||
                          index_constraints.field_constraint.inorder));
      if (!default_make_instance(ctx, target_memory, creation_constraints,
                                 instances.back(), TASK_MAPPING, force_new_instances,
                                 true/*meets*/, req, footprint))
        return false;
    }
  }
  // If we don't have anymore needed fields, we are done
  if (needed_fields.empty())
    return true;
  // There are no constraints for these fields so we get to do what we want
  instances.resize(instances.size()+1);
  LayoutConstraintSet creation_constraints = our_constraints;
  std::vector<FieldID> creation_fields;
  default_policy_select_instance_fields(ctx, req, needed_fields,
                                        creation_fields);
  creation_constraints.add_constraint(
      FieldConstraint(creation_fields, false/*contig*/, false/*inorder*/));
  if (!default_make_instance(ctx, target_memory, creation_constraints,
                             instances.back(), TASK_MAPPING, force_new_instances,
                             true/*meets*/,  req, footprint))
    return false;
  return true;
}


Legion::LayoutConstraintID TACOMapper::default_policy_select_layout_constraints(
    Legion::Mapping::MapperContext ctx, Legion::Memory target_memory,
    const Legion::RegionRequirement &req,
    MappingKind mapping_kind,
    bool needs_field_constraint_check,
    bool &force_new_instances) {
  std::cout << "In select layout constraints, need check=" << needs_field_constraint_check << std::endl;
  // Do something special for reductions and
  // it is not an explicit region-to-region copy
  if ((req.privilege == LEGION_REDUCE) && (mapping_kind != COPY_MAPPING))
  {
    // Always make new reduction instances
    force_new_instances = true;
    std::pair<Memory::Kind,ReductionOpID> constraint_key(
        target_memory.kind(), req.redop);
    std::map<std::pair<Memory::Kind,ReductionOpID>,LayoutConstraintID>::
    const_iterator finder = reduction_constraint_cache.find(
        constraint_key);
    // No need to worry about field constraint checks here
    // since we don't actually have any field constraints
    if (finder != reduction_constraint_cache.end())
      return finder->second;
    LayoutConstraintSet constraints;
    default_policy_select_constraints(ctx, constraints, target_memory, req);
    LayoutConstraintID result =
        runtime->register_layout(ctx, constraints);
    // Save the result
    reduction_constraint_cache[constraint_key] = result;
    return result;
  }
  // We always set force_new_instances to false since we are
  // deciding to optimize for minimizing memory usage instead
  // of avoiding Write-After-Read (WAR) dependences
  force_new_instances = false;
  // See if we've already made a constraint set for this layout
  std::pair<Memory::Kind,FieldSpace> constraint_key(target_memory.kind(),
                                                    req.region.get_field_space());
  std::map<std::pair<Memory::Kind,FieldSpace>,LayoutConstraintID>::
  const_iterator finder = layout_constraint_cache.find(constraint_key);
  if (finder != layout_constraint_cache.end() && false)
  {
    // If we don't need a constraint check we are already good
    if (!needs_field_constraint_check)
      return finder->second;
    // Check that the fields still are the same, if not, fall through
    // so that we make a new set of constraints
    const LayoutConstraintSet &old_constraints =
        runtime->find_layout_constraints(ctx, finder->second);
    // Should be only one unless things have changed
    const std::vector<FieldID> &old_set =
        old_constraints.field_constraint.get_field_set();
    // Check to make sure the field sets are still the same
    std::vector<FieldID> new_fields;
    runtime->get_field_space_fields(ctx,
                                    constraint_key.second,new_fields);
    if (new_fields.size() == old_set.size())
    {
      std::set<FieldID> old_fields(old_set.begin(), old_set.end());
      bool still_equal = true;
      for (unsigned idx = 0; idx < new_fields.size(); idx++)
      {
        if (old_fields.find(new_fields[idx]) == old_fields.end())
        {
          still_equal = false;
          break;
        }
      }
      if (still_equal)
        return finder->second;
    }
    // Otherwise we fall through and make a new constraint which
    // will also update the cache
  }
  // Fill in the constraints
  LayoutConstraintSet constraints;
  default_policy_select_constraints(ctx, constraints, target_memory, req);
  // Do the registration
  LayoutConstraintID result =
      runtime->register_layout(ctx, constraints);
  // Record our results, there is a benign race here as another mapper
  // call could have registered the exact same registration constraints
  // here if we were preempted during the registration call. The
  // constraint sets are identical though so it's all good.
  layout_constraint_cache[constraint_key] = result;
  return result;
}

void TACOMapper::default_map_task(const Legion::Mapping::MapperContext ctx,
                                  const Legion::Task &task,
                                  const MapTaskInput &input,
                                  MapTaskOutput &output) {
  Processor::Kind target_kind = task.target_proc.kind();
  // Get the variant that we are going to use to map this task
  VariantInfo chosen = default_find_preferred_variant(task, ctx,
                                                      true/*needs tight bound*/, true/*cache*/, target_kind);
  output.chosen_variant = chosen.variant;
  output.task_priority = default_policy_select_task_priority(ctx, task);
  output.postmap_task = false;
  // Figure out our target processors
  default_policy_select_target_processors(ctx, task, output.target_procs);
  Processor target_proc = output.target_procs[0];
  // See if we have an inner variant, if we do virtually map all the regions
  // We don't even both caching these since they are so simple
  if (chosen.is_inner) {
    // Check to see if we have any relaxed coherence modes in which
    // case we can no longer do virtual mappings so we'll fall through
    bool has_relaxed_coherence = false;
    for (unsigned idx = 0; idx < task.regions.size(); idx++) {
      if (task.regions[idx].prop != LEGION_EXCLUSIVE) {
        has_relaxed_coherence = true;
        break;
      }
    }
    if (!has_relaxed_coherence) {
      std::vector<unsigned> reduction_indexes;
      for (unsigned idx = 0; idx < task.regions.size(); idx++) {
        // As long as this isn't a reduction-only region requirement
        // we will do a virtual mapping, for reduction-only instances
        // we will actually make a physical instance because the runtime
        // doesn't allow virtual mappings for reduction-only privileges
        if (task.regions[idx].privilege == LEGION_REDUCE)
          reduction_indexes.push_back(idx);
        else
          output.chosen_instances[idx].push_back(
              PhysicalInstance::get_virtual_instance());
      }
      if (!reduction_indexes.empty()) {
        const TaskLayoutConstraintSet &layout_constraints =
            runtime->find_task_layout_constraints(ctx,
                                                  task.task_id, output.chosen_variant);
        for (std::vector<unsigned>::const_iterator it =
            reduction_indexes.begin(); it !=
                                       reduction_indexes.end(); it++) {
          MemoryConstraint mem_constraint =
              find_memory_constraint(ctx, task, output.chosen_variant, *it);
          Memory target_memory = default_policy_select_target_memory(ctx,
                                                                     target_proc,
                                                                     task.regions[*it],
                                                                     mem_constraint);
          std::set<FieldID> copy = task.regions[*it].privilege_fields;
          size_t footprint;
          if (!default_create_custom_instances(ctx, target_proc,
                                               target_memory, task.regions[*it], *it, copy,
                                               layout_constraints, false/*needs constraint check*/,
                                               output.chosen_instances[*it], &footprint)) {
            default_report_failed_instance_creation(task, *it,
                                                    target_proc, target_memory, footprint);
          }
        }
      }
      return;
    }
  }
  // Should we cache this task?
  CachedMappingPolicy cache_policy =
      default_policy_select_task_cache_policy(ctx, task);
  cache_policy = DEFAULT_CACHE_POLICY_DISABLE;

  // First, let's see if we've cached a result of this task mapping
  const unsigned long long task_hash = compute_task_hash(task);
  std::pair<TaskID, Processor> cache_key(task.task_id, target_proc);
  std::map<std::pair<TaskID, Processor>,
      std::list<CachedTaskMapping> >::const_iterator
      finder = cached_task_mappings.find(cache_key);
  // This flag says whether we need to recheck the field constraints,
  // possibly because a new field was allocated in a region, so our old
  // cached physical instance(s) is(are) no longer valid
  bool needs_field_constraint_check = true;
  if (cache_policy == DEFAULT_CACHE_POLICY_ENABLE && finder != cached_task_mappings.end()) {
    bool found = false;
    // Iterate through and see if we can find one with our variant and hash
    for (std::list<CachedTaskMapping>::const_iterator it =
        finder->second.begin(); it != finder->second.end(); it++) {
      if ((it->variant == output.chosen_variant) &&
          (it->task_hash == task_hash)) {
        // Have to copy it before we do the external call which
        // might invalidate our iterator
        output.chosen_instances = it->mapping;
        output.output_targets = it->output_targets;
        output.output_constraints = it->output_constraints;
        found = true;
        break;
      }
    }
    if (found) {
      // See if we can acquire these instances still
      if (runtime->acquire_and_filter_instances(ctx,
                                                output.chosen_instances))
        return;
      // We need to check the constraints here because we had a
      // prior mapping and it failed, which may be the result
      // of a change in the allocated fields of a field space
      needs_field_constraint_check = true;
      // If some of them were deleted, go back and remove this entry
      // Have to renew our iterators since they might have been
      // invalidated during the 'acquire_and_filter_instances' call
      default_remove_cached_task(ctx, output.chosen_variant,
                                 task_hash, cache_key, output.chosen_instances);
    }
  }
  // We didn't find a cached version of the mapping so we need to
  // do a full mapping, we already know what variant we want to use
  // so let's use one of the acceleration functions to figure out
  // which instances still need to be mapped.
  std::vector<std::set<FieldID> > missing_fields(task.regions.size());
  runtime->filter_instances(ctx, task, output.chosen_variant,
                            output.chosen_instances, missing_fields);
  // Track which regions have already been mapped
  std::vector<bool> done_regions(task.regions.size(), false);
  if (!input.premapped_regions.empty())
    for (std::vector<unsigned>::const_iterator it =
        input.premapped_regions.begin(); it !=
                                         input.premapped_regions.end(); it++)
      done_regions[*it] = true;
  const TaskLayoutConstraintSet &layout_constraints =
      runtime->find_task_layout_constraints(ctx,
                                            task.task_id, output.chosen_variant);
  // Now we need to go through and make instances for any of our
  // regions which do not have space for certain fields
  for (unsigned idx = 0; idx < task.regions.size(); idx++) {
    if (done_regions[idx])
      continue;
    // Skip any empty regions
    if ((task.regions[idx].privilege == LEGION_NO_ACCESS) ||
        (task.regions[idx].privilege_fields.empty()) ||
        missing_fields[idx].empty())
      continue;
    // See if this is a reduction
    MemoryConstraint mem_constraint =
        find_memory_constraint(ctx, task, output.chosen_variant, idx);
    Memory target_memory = default_policy_select_target_memory(ctx,
                                                               target_proc,
                                                               task.regions[idx],
                                                               mem_constraint);
    if (task.regions[idx].privilege == LEGION_REDUCE) {
      size_t footprint;
      if (!default_create_custom_instances(ctx, target_proc,
                                           target_memory, task.regions[idx], idx, missing_fields[idx],
                                           layout_constraints, needs_field_constraint_check,
                                           output.chosen_instances[idx], &footprint)) {
        default_report_failed_instance_creation(task, idx,
                                                target_proc, target_memory, footprint);
      }
      continue;
    }
    // Did the application request a virtual mapping for this requirement?
    if ((task.regions[idx].tag & DefaultMapper::VIRTUAL_MAP) != 0) {
      PhysicalInstance virt_inst = PhysicalInstance::get_virtual_instance();
      output.chosen_instances[idx].push_back(virt_inst);
      continue;
    }
    // Check to see if any of the valid instances satisfy this requirement
    {
      std::vector<PhysicalInstance> valid_instances;

      for (std::vector<PhysicalInstance>::const_iterator
               it = input.valid_instances[idx].begin(),
               ie = input.valid_instances[idx].end(); it != ie; ++it) {
        if (it->get_location() == target_memory)
          valid_instances.push_back(*it);
      }

      std::set<FieldID> valid_missing_fields;
      runtime->filter_instances(ctx, task, idx, output.chosen_variant,
                                valid_instances, valid_missing_fields);

#ifndef NDEBUG
      bool check =
#endif
          runtime->acquire_and_filter_instances(ctx, valid_instances);
      assert(check);

      output.chosen_instances[idx] = valid_instances;
      missing_fields[idx] = valid_missing_fields;

      if (missing_fields[idx].empty())
        continue;
    }
    // Otherwise make normal instances for the given region
    size_t footprint;
    if (!default_create_custom_instances(ctx, target_proc,
                                         target_memory, task.regions[idx], idx, missing_fields[idx],
                                         layout_constraints, needs_field_constraint_check,
                                         output.chosen_instances[idx], &footprint)) {
      default_report_failed_instance_creation(task, idx,
                                              target_proc, target_memory, footprint);
    }
  }

  // Finally we set a target memory for output instances
  Memory target_memory =
      default_policy_select_output_target(ctx, task.target_proc);
  for (unsigned i = 0; i < task.output_regions.size(); ++i) {
    output.output_targets[i] = target_memory;
    default_policy_select_output_constraints(
        task, output.output_constraints[i], task.output_regions[i]);
  }

  if (cache_policy == DEFAULT_CACHE_POLICY_ENABLE) {
    // Now that we are done, let's cache the result so we can use it later
    std::list<CachedTaskMapping> &map_list = cached_task_mappings[cache_key];
    map_list.push_back(CachedTaskMapping());
    CachedTaskMapping &cached_result = map_list.back();
    cached_result.task_hash = task_hash;
    cached_result.variant = output.chosen_variant;
    cached_result.mapping = output.chosen_instances;
    cached_result.output_targets = output.output_targets;
    cached_result.output_constraints = output.output_constraints;
  }
}

void TACOMapper::map_task(const Legion::Mapping::MapperContext ctx,
                          const Legion::Task &task,
                          const MapTaskInput &input,
                          MapTaskOutput &output) {
  // DefaultMapper::map_task(ctx, task, input, output);
  this->default_map_task(ctx, task, input, output);
  // If the tag is marked for untracked valid regions, then mark all of its
  // read only regions as up for collection.
  if ((task.tag & UNTRACK_VALID_REGIONS) != 0 && this->untrackValidRegions) {
    for (size_t i = 0; i < task.regions.size(); i++) {
      auto &rg = task.regions[i];
      if (rg.privilege == READ_ONLY) {
        output.untracked_valid_regions.insert(i);
      }
    }
  }
}

void TACOMapper::default_policy_select_constraints(Legion::Mapping::MapperContext ctx,
                                                   Legion::LayoutConstraintSet &constraints,
                                                   Legion::Memory target_memory,
                                                   const Legion::RegionRequirement &req) {
  // Ensure that regions are mapped in row-major order.
  Legion::IndexSpace is = req.region.get_index_space();
  Legion::Domain domain = runtime->get_index_space_domain(ctx, is);
  int dim = domain.get_dim();
  std::vector<Legion::DimensionKind> dimension_ordering(dim + 1);
  for (int i = 0; i < dim; ++i) {
    dimension_ordering[dim - i - 1] =
        static_cast<Legion::DimensionKind>(static_cast<int>(LEGION_DIM_X) + i);
  }
  dimension_ordering[dim] = LEGION_DIM_F;
  constraints.add_constraint(Legion::OrderingConstraint(dimension_ordering, false/*contiguous*/));

  // std::cout << std::hex << req.tag << " " << MAP_COLLECTIVE_INSTANCE << std::endl;
  if ((req.tag & MAP_COLLECTIVE_INSTANCE) != 0) {
    // Get a domain out from the projection arguments.
    Rect<2> bounds = *(Rect<2>*)(req.get_projection_args(NULL));
    Domain args(bounds);
    std::cout << "Requesting a collective instance for domain: " << args << std::endl;
    constraints.add_constraint(SpecializedConstraint(LEGION_AFFINE_SPECIALIZE, 0, false, false, args));
  }

  DefaultMapper::default_policy_select_constraints(ctx, constraints, target_memory, req);
}

void TACOMapper::default_policy_select_target_processors(
    Legion::Mapping::MapperContext ctx,
    const Legion::Task &task,
    std::vector<Legion::Processor> &target_procs) {
  // TODO (rohany): Add a TACO tag to the tasks.
  if (task.is_index_space) {
    // Index launches should be placed directly on the processor
    // they were sliced to.
    target_procs.push_back(task.target_proc);
  } else if (std::string(task.get_task_name()).find("task_") != std::string::npos) {
    // Other point tasks should stay on the originating processor, if they are
    // using a CPU Proc. Otherwise, send the tasks where the default mapper
    // says they should go. I think that the heuristics for OMP_PROC and TOC_PROC
    // are correct for our use case.
    if (task.target_proc.kind() == task.orig_proc.kind()) {
      target_procs.push_back(task.orig_proc);
    } else {
      DefaultMapper::default_policy_select_target_processors(ctx, task, target_procs);
    }
  } else {
    DefaultMapper::default_policy_select_target_processors(ctx, task, target_procs);
  }
}

Memory TACOMapper::default_policy_select_target_memory(Legion::Mapping::MapperContext ctx,
                                                       Legion::Processor target_proc,
                                                       const Legion::RegionRequirement &req,
                                                       Legion::MemoryConstraint mc) {
  // If we are supposed to perform NUMA aware allocations
  if (target_proc.kind() == Processor::OMP_PROC && this->numaAwareAllocs) {
    auto it = this->numaDomains.find(target_proc);
    assert(it != this->numaDomains.end());
    return it->second;
  } else {
    return DefaultMapper::default_policy_select_target_memory(ctx, target_proc, req, mc);
  }
}

int TACOMapper::default_policy_select_garbage_collection_priority(Legion::Mapping::MapperContext ctx, MappingKind kind,
                                                                  Legion::Memory memory,
                                                                  const Legion::Mapping::PhysicalInstance &instance,
                                                                  bool meets_fill_constraints, bool reduction) {
  // Copy the default mapper's heuristic to eagerly collection reduction instances.
  if (reduction) {
    return LEGION_GC_FIRST_PRIORITY;
  }
  // Deviate from the default mapper to give all instances default GC priority. The
  // default mapper most of the time marks instances as un-collectable from the GC,
  // which leads to problems when using instances in a "temporary buffer" style.
  return LEGION_GC_DEFAULT_PRIORITY;
}

std::vector<Legion::Processor> TACOMapper::select_targets_for_task(const Legion::Mapping::MapperContext ctx,
                                                       const Legion::Task& task) {
  auto kind = this->default_find_preferred_variant(task, ctx, false /* needs tight bounds */).proc_kind;
  // If we're requested to fill/validate on the CPU, then hijack the initial
  // processor selection to do so.
  if ((this->preferCPUFill && task.task_id == TID_TACO_FILL_TASK) ||
      (this->preferCPUValidate && task.task_id == TID_TACO_VALIDATE_TASK)) {
    // See if we have any OMP procs.
    auto targetKind = Legion::Processor::Kind::LOC_PROC;
    Legion::Machine::ProcessorQuery omps(this->machine);
    omps.only_kind(Legion::Processor::OMP_PROC);
    if (omps.count() > 0) {
      targetKind = Legion::Processor::Kind::OMP_PROC;
    }
    kind = targetKind;
  }
  // We always map to the same address space if replication is enabled.
  auto sameAddressSpace = ((task.tag & DefaultMapper::SAME_ADDRESS_SPACE) != 0) || this->replication_enabled;
  if (sameAddressSpace) {
    // If we are meant to stay local, then switch to return the appropriate
    // cached processors.
    switch (kind) {
      case Legion::Processor::OMP_PROC: {
        return this->local_omps;
      }
      case Legion::Processor::TOC_PROC: {
        return this->local_gpus;
      }
      case Legion::Processor::LOC_PROC: {
        return this->local_cpus;
      }
      default: {
        assert(false);
      }
    }
  } else {
    // If we are meant to distribute over all of the processors, then run a query
    // to find all processors of the desired kind.
    Legion::Machine::ProcessorQuery all_procs(machine);
    all_procs.only_kind(kind);
    return std::vector<Legion::Processor>(all_procs.begin(), all_procs.end());
  }

  // Keep the compiler happy.
  assert(false);
  return {};
}

void TACOMapper::slice_task(const Legion::Mapping::MapperContext ctx,
                            const Legion::Task &task,
                            const SliceTaskInput &input,
                            SliceTaskOutput &output) {
  if (task.tag & PLACEMENT) {
    // Placement tasks will put the dimensions of the placement grid at the beginning
    // of the task arguments. Here, we extract the packed placement grid dimensions.
    int dim = input.domain.get_dim();
    int *args = (int *) (task.args);
    std::vector<int> gridDims(dim);
    for (int i = 0; i < dim; i++) {
      gridDims[i] = args[i];
    }
    auto targets = this->select_targets_for_task(ctx, task);
    switch (dim) {
#define BLOCK(DIM) \
        case DIM:  \
          {        \
            Legion::DomainT<DIM, Legion::coord_t> pointSpace = input.domain; \
            this->decompose_points(pointSpace, gridDims, targets, output.slices);        \
            break;   \
          }
      LEGION_FOREACH_N(BLOCK)
#undef BLOCK
      default:
        assert(false);
    }
  } else {
    // Otherwise, we have our own implementation of slice task. The reason for this is
    // because the default mapper gets confused and hits a cache of domain slices. This
    // messes up the placement that we are going for with the index launches. This
    // implementation mirrors the standard slicing strategy of the default mapper.
    auto targets = this->select_targets_for_task(ctx, task);
    switch (input.domain.get_dim()) {
#define BLOCK(DIM) \
        case DIM:  \
          {        \
            Legion::DomainT<DIM,Legion::coord_t> point_space = input.domain; \
            Legion::Point<DIM,Legion::coord_t> num_blocks = \
              default_select_num_blocks<DIM>(targets.size(), point_space.bounds); \
            this->default_decompose_points<DIM>(point_space, targets, \
                  num_blocks, false/*recurse*/, \
                  stealing_enabled, output.slices); \
            break;   \
          }
      LEGION_FOREACH_N(BLOCK)
#undef BLOCK
      default:
        assert(false);
    }
  }
}
