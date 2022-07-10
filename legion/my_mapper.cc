/* Copyright 2022 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#define alignTo128Bytes false
#define SPARSE_INSTANCE false

#include "my_mapper.h"

#include "mappers/default_mapper.h"

#include "compiler/y.tab.c"
#include "compiler/lex.yy.c"

#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <fstream>

using namespace Legion;
using namespace Legion::Mapping;

static Logger log_mapper("nsmapper");

namespace Legion {
  namespace Internal {
    /**
     * \class UserShardingFunctor
     * The cyclic sharding functor just round-robins the points
     * onto the available set of shards
     */
    class UserShardingFunctor : public ShardingFunctor {
    private:
      std::string taskname;
    public:
      UserShardingFunctor(std::string mode);
      UserShardingFunctor(const UserShardingFunctor &rhs);
      virtual ~UserShardingFunctor(void);
    public:
      UserShardingFunctor& operator=(const UserShardingFunctor &rhs);
    public:
      virtual ShardID shard(const DomainPoint &point,
                            const Domain &full_space,
                            const size_t total_shards);
    };
  }
}

class NSMapper : public DefaultMapper
{
public:
  NSMapper(MapperRuntime *rt, Machine machine, Processor local, const char *mapper_name);

public:
  static std::string get_policy_file();
  void parse_policy_file(const std::string &policy_file);
  static void register_user_sharding_functors(Runtime *runtime);

private:
  Processor select_initial_processor_by_kind(const Task &task, Processor::Kind kind);
  bool validate_processor_mapping(MapperContext ctx, const Task &task, Processor proc, bool strict=true);
  template <typename Handle>
  void maybe_append_handle_name(const MapperContext ctx,
                                const Handle &handle,
                                std::vector<std::string> &names);
  void get_handle_names(const MapperContext ctx,
                        const RegionRequirement &req,
                        std::vector<std::string> &names);

public:
  virtual Processor default_policy_select_initial_processor(MapperContext ctx,
                                                            const Task &task);
  virtual void default_policy_select_target_processors(MapperContext ctx,
                                                       const Task &task,
                                                       std::vector<Processor> &target_procs);
  virtual LogicalRegion default_policy_select_instance_region(MapperContext ctx,
                                                              Memory target_memory,
                                                              const RegionRequirement &req,
                                                              const LayoutConstraintSet &constraints,
                                                              bool force_new_instances,
                                                              bool meets_constraints);
  virtual void map_task(const MapperContext ctx,
                        const Task &task,
                        const MapTaskInput &input,
                        MapTaskOutput &output);
  virtual void select_sharding_functor(
                                const MapperContext                ctx,
                                const Task&                        task,
                                const SelectShardingFunctorInput&  input,
                                      SelectShardingFunctorOutput& output);
  virtual void slice_task(const MapperContext      ctx,
                          const Task&              task,
                          const SliceTaskInput&    input,
                          SliceTaskOutput&   output);
  virtual void select_task_options(const MapperContext    ctx,
                                  const Task&            task,
                                        TaskOptions&     output);
  // copied from DISTAL/runtime/taco_mapper.cpp
  virtual void default_policy_select_constraints(Legion::Mapping::MapperContext ctx,
                                                   Legion::LayoutConstraintSet &constraints,
                                                   Legion::Memory target_memory,
                                                   const Legion::RegionRequirement &req);
protected:
  void custom_slice_task(const Task &task,
                          const std::vector<Processor> &local_procs,
                          const std::vector<Processor> &remote_procs,
                          const SliceTaskInput &input,
                               SliceTaskOutput &output) const;
  template<int DIM>
  static void custom_decompose_points(
                          const DomainT<DIM,coord_t> &point_space,
                          const std::vector<Processor> &targets,
                          bool recurse, bool stealable,
                          std::vector<TaskSlice> &slices, std::string taskname);

private:
  std::unordered_map<std::string, Processor::Kind> task_policies;
  std::unordered_map<TaskID, Processor::Kind> cached_task_policies;

  std::unordered_set<std::string> has_region_policy;
  using HashFn1 = PairHash<std::string, std::string>;
  std::unordered_map<std::pair<std::string, std::string>, Memory::Kind, HashFn1> region_policies;
  using HashFn2 = PairHash<TaskID, uint32_t>;
  std::unordered_map<std::pair<TaskID, uint32_t>, Memory::Kind, HashFn2> cached_region_policies;
  std::unordered_map<std::pair<TaskID, uint32_t>, std::string, HashFn2> cached_region_names;

  bool has_default_task_policy;
  std::vector<Processor::Kind> default_task_policy;

  std::unordered_map<Processor::Kind, std::vector<Memory::Kind>> default_region_policy;

public:
  static Tree2Legion tree_result;
  void set_task_policies(std::string, Processor::Kind);
  static std::unordered_map<std::string, ShardingID> task2sid;
};

Tree2Legion NSMapper::tree_result;
std::unordered_map<std::string, ShardingID> NSMapper::task2sid;

void NSMapper::set_task_policies(std::string x, Processor::Kind y)
{
  if (task_policies.count(x) > 0)
  {
    log_mapper.error() << "Duplicate " << x  << "'s processor mapping";
    assert(false);
  }
  task_policies.insert({x, y});
}

std::string NSMapper::get_policy_file()
{
  auto args = Runtime::get_input_args();
  for (auto idx = 0; idx < args.argc; ++idx)
  {
    if (strcmp(args.argv[idx], "-mapping") == 0)
    {
      if (idx + 1 >= args.argc) break;
      return args.argv[idx + 1];
    }
  }
  log_mapper.error("Policy file is missing");
  exit(-1);
}

std::string processor_kind_to_string(Processor::Kind kind)
{
  switch (kind)
  {
    case Processor::LOC_PROC: return "CPU";
    case Processor::TOC_PROC: return "GPU";
    case Processor::IO_PROC: return "IO";
    case Processor::PY_PROC: return "PY";
    case Processor::PROC_SET: return "PROC";
    case Processor::OMP_PROC: return "OMP";
    default:
    {
      assert(false);
      return "Unknown Kind";
    }
  }
}

std::string memory_kind_to_string(Memory::Kind kind)
{
  switch (kind)
  {
    case Memory::SYSTEM_MEM: return "SYSMEM";
    case Memory::GPU_FB_MEM: return "FBMEM";
    case Memory::REGDMA_MEM: return "RDMEM";
    case Memory::Z_COPY_MEM: return "ZCMEM";
    case Memory::SOCKET_MEM: return "SOCKETMEM";
    default:
    {
      assert(false);
      return "Unknown Kind";
    }
  }
}

void NSMapper::register_user_sharding_functors(Runtime *runtime)
{
  int i = 1;
  for (auto v: tree_result.task2func)
  {
    runtime->register_sharding_functor(i, new Legion::Internal::UserShardingFunctor(v.first));
    task2sid.insert({v.first, i});
    log_mapper.debug("%s inserted", v.first.c_str());
    i += 1;
  }
}

void NSMapper::parse_policy_file(const std::string &policy_file)
{
  log_mapper.debug("Policy file: %s", policy_file.c_str());
  tree_result = Tree2Legion(policy_file);
  if (tree_result.default_task_policy.size() > 0)
  {
    has_default_task_policy = true;
  }
  else
  {
    has_default_task_policy = false;
  }
  default_task_policy = tree_result.default_task_policy;
  default_region_policy = tree_result.default_region_policy;
  task_policies = tree_result.task_policies;
  region_policies = tree_result.region_policies;
  for (auto &v : region_policies)
  {
    has_region_policy.insert(v.first.first);
  }
}

Processor NSMapper::select_initial_processor_by_kind(const Task &task, Processor::Kind kind)
{
  Processor result;
  switch (kind)
  {
    case Processor::LOC_PROC:
    {
      result = local_cpus.front();
      break;
    }
    case Processor::TOC_PROC:
    {
      result = !local_gpus.empty() ? local_gpus.front() : local_cpus.front();
      break;
    }
    case Processor::IO_PROC:
    {
      result = !local_ios.empty() ? local_ios.front() : local_cpus.front();
      break;
    }
    case Processor::PY_PROC:
    {
      result = !local_pys.empty() ? local_pys.front() : local_cpus.front();
      break;
    }
    case Processor::PROC_SET:
    {
      result = !local_procsets.empty() ? local_procsets.front() : local_cpus.front();
      break;
    }
    case Processor::OMP_PROC:
    {
      result = !local_omps.empty() ? local_omps.front() : local_cpus.front();
      break;
    }
    default:
    {
      assert(false);
    }
  }

  auto kind_str = processor_kind_to_string(kind);
  if (result.kind() != kind)
  {
    log_mapper.warning(
      "Unsatisfiable policy: task %s requested %s, which does not exist",
      task.get_task_name(), kind_str.c_str());
  }
  else
  {
    log_mapper.debug(
      "Task %s is initially mapped to %s",
      task.get_task_name(), kind_str.c_str()
    );
  }
  return result;
}

bool NSMapper::validate_processor_mapping(MapperContext ctx, const Task &task, Processor proc, bool strict)
{
  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, task.task_id, variants, proc.kind());
  if (variants.empty())
  {
    if (strict)
    {
      auto kind_str = processor_kind_to_string(proc.kind());
      log_mapper.error(
        "Invalid policy: task %s requested %s, but has no valid task variant for the kind",
        task.get_task_name(), kind_str.c_str());
      exit(-1);
    }
    else
    {
      return false;
    }
  }
  return true;
}

Processor NSMapper::default_policy_select_initial_processor(MapperContext ctx, const Task &task)
{
  {
    auto finder = cached_task_policies.find(task.task_id);
    if (finder != cached_task_policies.end())
    {
      auto result = select_initial_processor_by_kind(task, finder->second);
      validate_processor_mapping(ctx, task, result);
      log_mapper.debug() << task.get_task_name() << " mapped by cache: " << processor_kind_to_string(result.kind()).c_str();
      return result;
    }
  }
  {
    auto finder = task_policies.find(task.get_task_name());
    if (finder != task_policies.end())
    {
      auto result = select_initial_processor_by_kind(task, finder->second);
      log_mapper.debug() << task.get_task_name() << " mapped by task_policies: " << processor_kind_to_string(result.kind()).c_str();
      validate_processor_mapping(ctx, task, result);
      cached_task_policies[task.task_id] = result.kind();
      return result;
    }
    else if (has_default_task_policy)
    {
      for (size_t i = 0; i < default_task_policy.size(); i++)
      {
        auto result = select_initial_processor_by_kind(task, default_task_policy[i]);
        if (result.kind() != default_task_policy[i])
        {
          log_mapper.debug("default_task_policy to map %s onto %s cannot satisfy, try next",
            task.get_task_name(), processor_kind_to_string(default_task_policy[i]).c_str());
          continue;
        }
        // default policy validation should not be strict, allowing fallback
        bool success = validate_processor_mapping(ctx, task, result, false);
        if (success)
        {
          log_mapper.debug() << task.get_task_name() << "  mapped by default_task_policy: " << processor_kind_to_string(result.kind()).c_str();
          cached_task_policies[task.task_id] = result.kind();
          return result;
        }
        else
        {
          log_mapper.debug("default_task_policy to map %s onto %s cannot satisfy, try next",
            task.get_task_name(), processor_kind_to_string(default_task_policy[i]).c_str());
        }
      }
    }
    if (has_default_task_policy)
    {
      log_mapper.debug(
        "None of the user-specified default processors work for task %s, fall back",
        task.get_task_name());
    }
  }
  log_mapper.debug(
    "No processor policy is given for task %s, falling back to the default policy",
    task.get_task_name());
  return DefaultMapper::default_policy_select_initial_processor(ctx, task);
}

void NSMapper::default_policy_select_target_processors(MapperContext ctx,
                                                       const Task &task,
                                                       std::vector<Processor> &target_procs)
{
  target_procs.push_back(task.target_proc);
}

LogicalRegion NSMapper::default_policy_select_instance_region(MapperContext ctx,
                                                              Memory target_memory,
                                                              const RegionRequirement &req,
                                                              const LayoutConstraintSet &constraints,
                                                              bool force_new_instances,
                                                              bool meets_constraints)
{
  return req.region;
}

template <typename Handle>
void NSMapper::maybe_append_handle_name(const MapperContext ctx,
                                        const Handle &handle,
                                        std::vector<std::string> &names)
{
  const void *result = nullptr;
  size_t size = 0;
  if (runtime->retrieve_semantic_information(
        ctx, handle, LEGION_NAME_SEMANTIC_TAG, result, size, true, true))
    names.push_back(std::string(static_cast<const char*>(result)));
}

void NSMapper::get_handle_names(const MapperContext ctx,
                                const RegionRequirement &req,
                                std::vector<std::string> &names)
{
  maybe_append_handle_name(ctx, req.region, names);

  if (runtime->has_parent_logical_partition(ctx, req.region))
  {
    auto parent = runtime->get_parent_logical_partition(ctx, req.region);
    maybe_append_handle_name(ctx, parent, names);
  }

  if (req.region != req.parent)
    maybe_append_handle_name(ctx, req.parent, names);
}

void NSMapper::map_task(const MapperContext      ctx,
                        const Task&              task,
                        const MapTaskInput&      input,
                              MapTaskOutput&     output)
{
  Processor::Kind target_proc_kind = task.target_proc.kind();
  
  if (has_region_policy.find(task.get_task_name()) == has_region_policy.end() 
      && default_region_policy.find(target_proc_kind) == default_region_policy.end())
  {
    log_mapper.debug(
      "No memory policy is given for task %s, falling back to the default policy",
      task.get_task_name());
    DefaultMapper::map_task(ctx, task, input, output);
    return;
  }

  VariantInfo chosen = default_find_preferred_variant(task, ctx,
                    true/*needs tight bound*/, true/*cache*/, target_proc_kind);
  output.chosen_variant = chosen.variant;
  output.task_priority = default_policy_select_task_priority(ctx, task);
  output.postmap_task = false;
  default_policy_select_target_processors(ctx, task, output.target_procs);
  log_mapper.debug("map_task for selecting memory will use %s as processor", 
    processor_kind_to_string(output.target_procs[0].kind()).c_str());

  if (chosen.is_inner)
  {
    log_mapper.debug(
      "is_inner = true; Unsupported variant is chosen for task %s, falling back to the default policy",
      task.get_task_name());
    DefaultMapper::map_task(ctx, task, input, output);
    return;
  }

  const TaskLayoutConstraintSet &layout_constraints =
    runtime->find_task_layout_constraints(ctx, task.task_id, output.chosen_variant);

  for (uint32_t idx = 0; idx < task.regions.size(); ++idx)
  {
    auto &req = task.regions[idx];
    if (req.privilege == LEGION_NO_ACCESS || req.privilege_fields.empty()) continue;

    bool found_policy = false;
    Memory::Kind target_kind = Memory::SYSTEM_MEM;
    Memory target_memory = Memory::NO_MEMORY;
    std::string region_name;

    auto cache_key = std::make_pair(task.task_id, idx);
    auto finder = cached_region_policies.find(cache_key);
    if (finder != cached_region_policies.end())
    {
      found_policy = true;
      target_kind = finder->second;
      region_name = cached_region_names.find(cache_key)->second;
    }

    if (!found_policy)
    {
      std::vector<std::string> path;
      get_handle_names(ctx, req, path);
      for (auto &name : path)
      {
        auto finder = region_policies.find(std::make_pair(task.get_task_name(), name));
        if (finder != region_policies.end())
        {
          target_kind = finder->second;
          found_policy = true;
        }
        else if (default_region_policy.find(target_proc_kind) != default_region_policy.end())
        {
          for (size_t regidx = 0; regidx < default_region_policy[target_proc_kind].size(); regidx++)
          {
            auto target_kind_cand = default_region_policy[target_proc_kind][regidx];
            Machine::MemoryQuery visible_memories(machine);
            visible_memories.has_affinity_to(task.target_proc);
            visible_memories.only_kind(target_kind_cand);
            if (visible_memories.count() > 0)
            {
              target_memory = visible_memories.first();
              if (target_memory.exists())
              {
                target_kind = target_kind_cand;
                found_policy = true;
                break;
              }
            }
          }
        }
        else
        {
          assert(false);
        }
        auto key = std::make_pair(task.task_id, idx);
        cached_region_policies[key] = target_kind;
        cached_region_names[key] = name;
        region_name = name;
        break;
      }
    }

    if (found_policy)
    {
      Machine::MemoryQuery visible_memories(machine);
      visible_memories.has_affinity_to(task.target_proc);
      visible_memories.only_kind(target_kind);
      if (visible_memories.count() > 0)
        target_memory = visible_memories.first();
    }

    if (target_memory.exists())
    {
      auto kind_str = memory_kind_to_string(target_kind);
      log_mapper.debug(
          "Region %u of task %s (%s) is mapped to %s",
          idx, task.get_task_name(), region_name.c_str(), kind_str.c_str());
    }
    else
    {
      log_mapper.debug(
        "Unsatisfiable policy for memory: region %u of task %s cannot be mapped to %s, falling back to the default policy",
        idx, task.get_task_name(), memory_kind_to_string(target_kind).c_str());
      auto mem_constraint =
        find_memory_constraint(ctx, task, output.chosen_variant, idx);
      target_memory =
        default_policy_select_target_memory(ctx, task.target_proc, req, mem_constraint);
    }

    auto missing_fields = req.privilege_fields;
    if (req.privilege == LEGION_REDUCE)
    {
      size_t footprint;
      if (!default_create_custom_instances(ctx, task.target_proc,
              target_memory, req, idx, missing_fields,
              layout_constraints, true,
              output.chosen_instances[idx], &footprint))
      {
        default_report_failed_instance_creation(task, idx,
              task.target_proc, target_memory, footprint);
      }
      continue;
    }

    std::vector<PhysicalInstance> valid_instances;

    for (auto &instance : input.valid_instances[idx])
      if (instance.get_location() == target_memory)
        valid_instances.push_back(instance);

    runtime->filter_instances(ctx, task, idx, output.chosen_variant,
                              valid_instances, missing_fields);

    bool check = runtime->acquire_and_filter_instances(ctx, valid_instances);
    assert(check);

    output.chosen_instances[idx] = valid_instances;

    if (missing_fields.empty()) continue;

    size_t footprint;
    if (!default_create_custom_instances(ctx, task.target_proc,
            target_memory, req, idx, missing_fields,
            layout_constraints, true,
            output.chosen_instances[idx], &footprint))
    {
      default_report_failed_instance_creation(task, idx,
              task.target_proc, target_memory, footprint);
    }
  }
}

void NSMapper::select_sharding_functor(
                                const MapperContext                ctx,
                                const Task&                        task,
                                const SelectShardingFunctorInput&  input,
                                      SelectShardingFunctorOutput& output)
{
  auto finder = task2sid.find(task.get_task_name());
  if (finder != task2sid.end())
  {
    output.chosen_functor = finder->second;
    log_mapper.debug("select_sharding_functor for task %s: %d",
      task.get_task_name(), output.chosen_functor);
  }
  else
  {
    log_mapper.debug("No sharding functor found in select_sharding_functor %s, fall back to default", task.get_task_name());
    output.chosen_functor = 0; // default functor
  }
}

void NSMapper::select_task_options(const MapperContext    ctx,
                                  const Task&            task,
                                        TaskOptions&     output)
//--------------------------------------------------------------------------
{
  log_mapper.debug("NSMapper select_task_options in %s", get_mapper_name());
  output.initial_proc = default_policy_select_initial_processor(ctx, task);
  output.inline_task = false;
  output.stealable = stealing_enabled;
  // This is the best choice for the default mapper assuming
  // there is locality in the remote mapped tasks
  output.map_locally = map_locally;
  // Control replicate the top-level task in multi-node settings
  // otherwise we do no control replication
#ifdef DEBUG_CTRL_REPL
  if (task.get_depth() == 0)
#else
  if ((total_nodes > 1) && (task.get_depth() == 0))
#endif
    output.replicate = replication_enabled;
    //output.replicate = false; // no replication for now..
  else
    output.replicate = false;
}

template<int DIM>
/*static*/ void NSMapper::custom_decompose_points(
                        const DomainT<DIM,coord_t> &point_space,
                        const std::vector<Processor> &targets,
                        bool recurse, bool stealable,
                        std::vector<TaskSlice> &slices,
                        std::string taskname)
//--------------------------------------------------------------------------
{
  log_mapper.debug() << "custom_decompose_points, dim=" << DIM 
    << " point_space.volume()=" << point_space.volume()
    << " point_space=[" << point_space.bounds.lo[0] << "," << point_space.bounds.hi[0] << "]";
  slices.reserve(point_space.volume());
  
  for (Realm::IndexSpaceIterator<DIM,coord_t> it(point_space); it.valid; it.step()) 
  {
    for (Legion::PointInRectIterator<DIM,coord_t> itr(it.rect); itr(); itr++)
    {
      const Point<DIM,coord_t> point = *itr;
      // todo: use function's computation results
      std::vector<int> index_point;
      log_mapper.debug("slice point: ");
      for (int i = 0; i < DIM; i++)
      {
        index_point.push_back(point[i]);
        log_mapper.debug() << point[i] << " ,";
      }
      size_t slice_res = (size_t) tree_result.run(taskname, index_point)[1];
      log_mapper.debug("--> %ld", slice_res);
      if (slice_res >= targets.size())
      {
        log_mapper.error("%ld >= %ld, targets out of bound!", slice_res, targets.size());
        assert(false);
      }
      // Construct the output slice for Legion.
      Legion::DomainT<DIM, Legion::coord_t> slice;
      slice.bounds.lo = point;
      slice.bounds.hi = point;
      slice.sparsity = point_space.sparsity;
      if (!slice.dense()) { slice = slice.tighten(); }
      if (slice.volume() > 0) 
      {
        TaskSlice ts;
        ts.domain = slice;
        ts.proc = targets[slice_res];
        ts.recurse = recurse;
        ts.stealable = stealable;
        slices.push_back(ts);
      }
    }
  }
}


void NSMapper::custom_slice_task(const Task &task,
                                const std::vector<Processor> &local,
                                const std::vector<Processor> &remote,
                                const SliceTaskInput& input,
                                      SliceTaskOutput &output) const
  //--------------------------------------------------------------------------
{
  // The two-level decomposition doesn't work so for now do a
  // simple one-level decomposition across all the processors.
  Machine::ProcessorQuery all_procs(machine);
  all_procs.only_kind(local[0].kind());
  all_procs.local_address_space();
  size_t node_num = Machine::get_machine().get_address_space_count();
  log_mapper.debug("how many nodes? %ld", node_num);
  log_mapper.debug("how many processors? local=%ld, remote=%ld", local.size(), remote.size());
  log_mapper.debug("node_id = %d", node_id);
  // if ((task.tag & SAME_ADDRESS_SPACE) != 0 || same_address_space)
  // {
  //   log_mapper.debug("local_address_space executed");
  //   all_procs.local_address_space();
  // }
  std::vector<Processor> procs(all_procs.begin(), all_procs.end());
  log_mapper.debug("Inside custom_slice_task for %s, procs=%ld, dim=%d",
    task.get_task_name(), procs.size(), input.domain.get_dim());
  
  // todo: use task.get_task_name() to find the slicing function to invoke
  switch (input.domain.get_dim())
  {
#define BLOCK(DIM) \
    case DIM: \
      { \
        DomainT<DIM,coord_t> point_space = input.domain; \
        custom_decompose_points<DIM>(point_space, procs, \
              false/*recurse*/, stealing_enabled, output.slices, task.get_task_name()); \
        break; \
      }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default: // don't support other dimensions right now
      assert(false);
  }
}

void NSMapper::slice_task(const MapperContext      ctx,
                          const Task&              task, 
                          const SliceTaskInput&    input,
                          SliceTaskOutput&   output)
{
  if (tree_result.task2func.count(std::string(task.get_task_name())) == 0)
  {
    log_mapper.debug("Use default slice_task for %s", task.get_task_name());
    DefaultMapper::slice_task(ctx, task, input, output);
    return;
  }
  log_mapper.debug("Customize slice_task for %s", task.get_task_name());
  // Whatever kind of processor we are is the one this task should
  // be scheduled on as determined by select initial task
  Processor::Kind target_kind =
    task.must_epoch_task ? local_proc.kind() : task.target_proc.kind();
  log_mapper.debug("%d,%d:%d", target_kind, local_proc.kind(), task.target_proc.kind());
  switch (target_kind)
  {
    case Processor::LOC_PROC:
      {
        log_mapper.debug("%d: CPU here", target_kind);
        custom_slice_task(task, local_cpus, remote_cpus, input, output);
        break;
      }
    case Processor::TOC_PROC:
      {
        log_mapper.debug("%d: GPU here", target_kind);
        custom_slice_task(task, local_gpus, remote_gpus, input, output);
        break;
      }
    case Processor::IO_PROC:
      {
        log_mapper.debug("%d: IO here", target_kind);
        custom_slice_task(task, local_ios, remote_ios, input, output);
        break;
      }
    case Processor::PY_PROC:
      {
        log_mapper.debug("%d: PY here", target_kind);
        custom_slice_task(task, local_pys, remote_pys, input, output);
        break;
      }
    case Processor::PROC_SET:
      {
        log_mapper.debug("%d: PROC here", target_kind);
        custom_slice_task(task, local_procsets, remote_procsets, input, output);
        break;
      }
    case Processor::OMP_PROC:
      {
        log_mapper.debug("%d: OMP here", target_kind);
        custom_slice_task(task, local_omps, remote_omps, input, output);
        break;
      }
    default:
      assert(false); // unimplemented processor kind
  }
  // const Rect<1> bounds = input.domain; 
  // const size_t num_points = bounds.volume();
  // output.slices.reserve(num_points);
  // log_mapper.debug("In slice_task");
  // if (num_points == local_pys.size())
  // {
  //   log_mapper.debug("sharded, %ld, %ld", num_points, local_pys.size());
  //   unsigned index = 0;
  //   // Already been sharded, just assign to the local python procs
  //   for (coord_t p = bounds.lo[0]; p <= bounds.hi[0]; p++)
  //   {
  //     const Point<1> point(p);
  //     const Rect<1> rect(point,point);
  //     output.slices.push_back(TaskSlice(Domain(rect),
  //           local_pys[index++], false/*recurse*/, false/*stelable*/));
  //   }
  // }
  // else
  // {
  //   log_mapper.debug("not sharded, %ld, %ld, %d", num_points, local_pys.size(), total_nodes);
  //   // Not sharded, so we should have points for all the python procs
  //   assert(input.domain.get_volume() == (local_pys.size() * total_nodes));
  //   Machine::ProcessorQuery py_procs(machine);
  //   py_procs.only_kind(Processor::PY_PROC);
  //   std::set<AddressSpaceID> spaces;
  //   for (Machine::ProcessorQuery::iterator it = 
  //         py_procs.begin(); it != py_procs.end(); it++)
  //   {
  //     const AddressSpaceID space = it->address_space();
  //     if (spaces.find(space) != spaces.end())
  //       continue;
  //     const Point<1> lo(space*local_pys.size());
  //     const Point<1> hi((space+1)*local_pys.size()-1);
  //     const Rect<1> rect(lo,hi);
  //     output.slices.push_back(TaskSlice(Domain(rect),
  //           *it, true/*recurse*/, false/*stelable*/));
  //     spaces.insert(space);
  //   }
  // }
}

NSMapper::NSMapper(MapperRuntime *rt, Machine machine, Processor local, const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
{
  std::string policy_file = get_policy_file();
  parse_policy_file(policy_file);
}

static void create_mappers(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs)
{
  log_mapper.debug("Inside create_mappers local_procs.size() = %ld", local_procs.size());
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    NSMapper* mapper = new NSMapper(runtime->get_mapper_runtime(), machine, *it, "ns_mapper");
    if (it == local_procs.begin())
    {
      NSMapper::register_user_sharding_functors(runtime);
    }
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  Runtime::add_registration_callback(create_mappers);
}

namespace Legion {
  namespace Internal {
        /////////////////////////////////////////////////////////////
    // User Sharding Functor
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    UserShardingFunctor::UserShardingFunctor(std::string mode)
      : ShardingFunctor(), taskname(mode)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    UserShardingFunctor::UserShardingFunctor(
                                            const UserShardingFunctor &rhs)
      : ShardingFunctor()
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    UserShardingFunctor::~UserShardingFunctor(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    UserShardingFunctor& UserShardingFunctor::operator=(
                                              const UserShardingFunctor &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    ShardID UserShardingFunctor::shard(const DomainPoint &point,
                                         const Domain &full_space,
                                         const size_t total_shards)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(point.get_dim() == full_space.get_dim());
#endif
      log_mapper.debug("shard dim: %d, total_shards: %ld", point.get_dim(), total_shards);
      // size_t node_num = Machine::get_machine().get_address_space_count();
      switch (point.get_dim())
      {
        // TODO: link with AST
#define DIMFUNC(DIM) \
        case DIM: \
          { \
            const DomainT<DIM,coord_t> is = full_space; \
            const Point<DIM,coord_t> p1 = point; \
            std::vector<int> index_point, launch_space; \
            for (int i = 0; i < DIM; i++) \
            { \
              index_point.push_back(p1[i]); \
              launch_space.push_back(is.bounds.hi[i] - is.bounds.lo[i] + 1); \
            } \
            NSMapper::tree_result.set_launch_space(launch_space); \
            log_mapper.debug("shard point: "); \
            for (int i = 0; i < DIM; i++) \
            { \
              log_mapper.debug("%lld, ", point[i]); \
            } \
            log_mapper.debug(" --> node %d\n", (int) NSMapper::tree_result.run(taskname, index_point)[0]); \
            return NSMapper::tree_result.run(taskname, index_point)[0]; \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
      log_mapper.debug("shard: should never reach");
      assert(false);
      return 0;
    }
  }
}


// copied from DISTAL/runtime/taco_mapper.cpp
// add #define for undeclared fields

void NSMapper::default_policy_select_constraints(Legion::Mapping::MapperContext ctx,
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
  // If we were requested to have an alignment, add the constraint.
  if (/*this->*/alignTo128Bytes) {
    for (auto it : req.privilege_fields) {
      constraints.add_constraint(Legion::AlignmentConstraint(it, LEGION_EQ_EK, 128));
    }
  }
  // If the instance is supposed to be sparse, tell Legion we want it that way.
  // Unfortunately because we are adjusting the SpecializedConstraint, we have to
  // fully override the default mapper because there appears to be some undefined
  // behavior when two specialized constraints are added.
  if ((req.tag & SPARSE_INSTANCE) != 0) {
    taco_iassert(req.privilege != LEGION_REDUCE);
    constraints.add_constraint(SpecializedConstraint(LEGION_COMPACT_SPECIALIZE));
  } else if (req.privilege == LEGION_REDUCE) {
    // Make reduction fold instances.
    constraints.add_constraint(SpecializedConstraint(
                        LEGION_AFFINE_REDUCTION_SPECIALIZE, req.redop))
      .add_constraint(MemoryConstraint(target_memory.kind()));
  } else {
    // Our base default mapper will try to make instances of containing
    // all fields (in any order) laid out in SOA format to encourage
    // maximum re-use by any tasks which use subsets of the fields
    constraints.add_constraint(SpecializedConstraint())
               .add_constraint(MemoryConstraint(target_memory.kind()));
    if (constraints.field_constraint.field_set.size() == 0)
    {
      // Normal instance creation
      std::vector<FieldID> fields;
      default_policy_select_constraint_fields(ctx, req, fields);
      constraints.add_constraint(FieldConstraint(fields,false/*contiguous*/,false/*inorder*/));
    }
  }
}
