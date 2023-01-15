Task * GPU,OMP,CPU;

Region * * * SOCKMEM,SYSMEM;
Region * * GPU FBMEM,ZCMEM;

# Layout taskname regionname memory AOS F_order;
Layout * * * SOA C_order; # Align==128 Compact

CollectMemory task_4 *;

def block_primitive(IPoint x, ISpace y, MSpace z, int dim1, int dim2) {
    return x[dim1] * z.size[dim2] / y.size[dim1];
}

def cyclic_primitive(IPoint x, ISpace y, MSpace z, int dim1, int dim2) {
    return x[dim1] % z.size[dim2];
}

m_2d = Machine(GPU); # (nodes * processors) * 1 due to multiple ($GPUs_per_node) ranks per node

def autoblock(Task task) {
    m_4d = m_2d.auto_split(0, task.ispace);
    upper = tuple(block_primitive(task.ipoint, task.ispace, m_4d, i, i) for i in (0, 1, 2));
    return m_4d[*upper, 0];
}

def block_zero(Task task) {
    shard_dim = block_primitive(task.ipoint, task.ispace, m_2d, 0, 0);
    return m_2d[shard_dim, 0];
}

def placement_shard(Task task) {
    rank_size = m_2d.size[0];
    grid_size = task.ispace[0] > task.ispace[2] ? task.ispace[0] : task.ispace[2];
    linearize = task.ipoint[0] + task.ipoint[1] * grid_size + task.ipoint[2] * grid_size * grid_size;
    return m_2d[linearize % rank_size, 0];
}

def init_all_gpus(Task task) {
    return m_2d[task.ipoint[0], 0];
}

# PLACEMENT_SHARD (grid, grid, grid):
# task_1 (grid, grid, 1)
# task_2 (grid, 1, grid)
# task_3 (1, grid, grid)
IndexTaskMap task_1,task_2,task_3 placement_shard;
# task_4 (grid, grid, grid), UNTRACK_VALID_REGIONS
IndexTaskMap task_4 autoblock;
# init_cublas 1-dim launch for ALL GPUs
IndexTaskMap init_cublas init_all_gpus;