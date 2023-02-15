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

m_2d = Machine(GPU); # nodes * processors

def autoblock(Task task) {
    m_4d = m_2d.auto_split(0, task.ispace);
    upper = tuple(block_primitive(task.ipoint, task.ispace, m_4d, i, i) for i in (0, 1, 2));
    return m_4d[*upper, 0];
}

def placement_shard(Task task) {
    rank_size = m_2d.size[0];
    m_5d = m_2d.auto_split(0, (1, 1, 1)); # evenly split
    gx = m_5d.size[2];
    gy = m_5d.size[1];
    gz = m_5d.size[0];
    # print("gx={}, gy={}, gz={}", gx, gy, gz);
    linearize = task.ipoint[0] + task.ipoint[1] * gx + task.ipoint[2] * gx * gy;
    return m_2d[linearize % rank_size, 0];
}

def init_all_gpus(Task task) {
    return m_2d[task.ipoint[0], 0];
}

# PLACEMENT_SHARD (gx, gy, gz):
# task_1 (gx, gy, 1)
# task_2 (gx, 1, gz)
# task_3 (1, gy, gz)
# IndexTaskMap task_1,task_2,task_3 placement_shard;
# task_4 (gx, gy, gz), UNTRACK_VALID_REGIONS
IndexTaskMap task_4 autoblock;
# init_cublas 1-dim launch for ALL GPUs
IndexTaskMap init_cublas init_all_gpus;
