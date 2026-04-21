use std::sync::Arc;
use std::time::Instant;

use diskann::{
    graph::{
        AdjacencyList, DiskANNIndex, InplaceDeleteMethod,
        test::{provider},
    },
    provider::{Delete, NeighborAccessor},
    utils::ONE,
};
use diskann_benchmark_core::{
    build::{
        self,
        graph::{MultiInsert, SingleInsert},
        ids,
    },
    streaming::graph::{DropDeleted, InplaceDelete},
};

const DATASET_PATH: &str = "/Users/priya/Downloads/big-ann-benchmarks/data/random10000/data_10000_20";
const MAX_DEGREE: usize = 32;

fn load_dataset(path: &str) -> Arc<diskann_utils::views::Matrix<f32>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).expect("Failed to open dataset file");
    let mut buf = [0u8; 4];

    file.read_exact(&mut buf).unwrap();
    let num_vectors = u32::from_le_bytes(buf) as usize;

    file.read_exact(&mut buf).unwrap();
    let dim = u32::from_le_bytes(buf) as usize;

    println!("Loading dataset: {} vectors, {} dimensions", num_vectors, dim);

    let total = num_vectors * dim;
    let mut flat: Vec<f32> = vec![0.0f32; total];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut flat);
    file.read_exact(bytes).unwrap();
    Arc::new(diskann_utils::views::Matrix::try_from(flat.into_boxed_slice(), num_vectors, dim).unwrap())
}

fn build_index_from_data(
    data: Arc<diskann_utils::views::Matrix<f32>>,
) -> (
    Arc<DiskANNIndex<provider::Provider>>,
    usize,
) {
    let start_id = u32::MAX;
    let distance = diskann_vector::distance::Metric::L2;
    let nrows = data.nrows();
    let dim = data.ncols();

    // Use the first vector as the start point
    let start_point = data.row(0).to_vec();

    let provider_config = provider::Config::new(
        distance,
        MAX_DEGREE + 4,
        std::iter::once(provider::StartPoint::new(start_id, start_point)),
    )
    .unwrap();

    let provider = provider::Provider::new(provider_config);

    let index_config = diskann::graph::config::Builder::new(
        provider.max_degree().checked_sub(3).unwrap(),
        diskann::graph::config::MaxDegree::new(provider.max_degree()),
        64,
        distance.into(),
    )
    .build()
    .unwrap();

    let index = Arc::new(DiskANNIndex::new(index_config, provider, None));

    (index, nrows)
}

fn print_degree_stats(
    label: &str,
    index: &Arc<DiskANNIndex<provider::Provider>>,
    nrows: usize,
    rt: &tokio::runtime::Runtime,
) {
    let accessor = index.provider().neighbors();
    let mut adj = AdjacencyList::new();
    let mut max_degree = 0usize;
    let mut min_degree = usize::MAX;
    let mut total = 0usize;
    let mut cnt_less_than_two = 0usize;

    for i in 0..nrows as u32 {
        rt.block_on(accessor.get_neighbors(i, &mut adj)).unwrap();
        let deg = adj.len();
        max_degree = max_degree.max(deg);
        min_degree = min_degree.min(deg);
        total += deg;
        if deg < 2 {
            cnt_less_than_two += 1;
        }
    }

    let avg = total as f32 / nrows as f32;
    println!(
        "[{}] Degree stats -> max: {}, min: {}, avg: {:.2}, nodes with degree < 2: {}",
        label, max_degree, min_degree, avg, cnt_less_than_two
    );
}

fn test_static_index(data: Arc<diskann_utils::views::Matrix<f32>>) {
    println!("\n=== Static Index ===");

    let (index, nrows) = build_index_from_data(data.clone());
    let rt = diskann_benchmark_core::tokio::runtime(1).unwrap();

    let start = Instant::now();
    build::build(
        SingleInsert::new(
            index.clone(),
            data.clone(),
            provider::Strategy::new(),
            ids::Identity::<u32>::new(),
        ),
        build::Parallelism::dynamic(ONE, ONE),
        &rt,
    )
    .unwrap();
    let elapsed = start.elapsed();

    println!("Built static index with {} vectors in {:.2?}", nrows, elapsed);
    println!(
        "Throughput: {:.0} inserts/sec",
        nrows as f64 / elapsed.as_secs_f64()
    );
    print_degree_stats("static after build", &index, nrows, &rt);
    println!("Static index: no insert/delete after build");
}

fn test_dynamic_single_insert(data: Arc<diskann_utils::views::Matrix<f32>>) {
    println!("\n=== Dynamic Index: Single Insert ===");

    let (index, nrows) = build_index_from_data(data.clone());
    let rt = diskann_benchmark_core::tokio::runtime(1).unwrap();

    let start = Instant::now();
    build::build(
        SingleInsert::new(
            index.clone(),
            data.clone(),
            provider::Strategy::new(),
            ids::Identity::<u32>::new(),
        ),
        build::Parallelism::dynamic(ONE, ONE),
        &rt,
    )
    .unwrap();
    let elapsed = start.elapsed();

    println!("Inserted {} nodes one by one in {:.2?}", nrows, elapsed);
    println!(
        "Throughput: {:.0} inserts/sec",
        nrows as f64 / elapsed.as_secs_f64()
    );
    print_degree_stats("single insert after build", &index, nrows, &rt);
}

fn test_dynamic_multi_insert(data: Arc<diskann_utils::views::Matrix<f32>>) {
    println!("\n=== Dynamic Index: Multi Insert ===");

    let (index, nrows) = build_index_from_data(data.clone());
    let rt = diskann_benchmark_core::tokio::runtime(2).unwrap();

    let start = Instant::now();
    build::build(
        MultiInsert::new(
            index.clone(),
            data.clone(),
            provider::Strategy::new(),
            ids::Identity::<u32>::new(),
        ),
        build::Parallelism::dynamic(ONE, std::num::NonZeroUsize::new(2).unwrap()),
        &rt,
    )
    .unwrap();
    let elapsed = start.elapsed();

    println!("Multi-inserted {} nodes in {:.2?}", nrows, elapsed);
    println!(
        "Throughput: {:.0} inserts/sec",
        nrows as f64 / elapsed.as_secs_f64()
    );
    print_degree_stats("multi insert after build", &index, nrows, &rt);
}

fn test_dynamic_inplace_delete(data: Arc<diskann_utils::views::Matrix<f32>>) {
    println!("\n=== Dynamic Index: Inplace Delete ===");

    let (index, nrows) = build_index_from_data(data.clone());
    let rt = diskann_benchmark_core::tokio::runtime(2).unwrap();

    build::build(
        SingleInsert::new(
            index.clone(),
            data.clone(),
            provider::Strategy::new(),
            ids::Identity::<u32>::new(),
        ),
        build::Parallelism::dynamic(ONE, ONE),
        &rt,
    )
    .unwrap();

    println!("Built index with {} nodes", nrows);
    print_degree_stats("before delete", &index, nrows, &rt);

    let to_delete: Box<[u32]> = Box::new([0u32, 1, 2, 3, 4]);
    println!("Deleting nodes: {:?}", to_delete);

    let start = Instant::now();
    build::build(
        InplaceDelete::new(
            index.clone(),
            provider::Strategy::new(),
            4,
            InplaceDeleteMethod::TwoHopAndOneHop,
            ids::Slice::new(to_delete.clone()),
        ),
        build::Parallelism::dynamic(ONE, std::num::NonZeroUsize::new(2).unwrap()),
        &rt,
    )
    .unwrap();
    let elapsed = start.elapsed();

    println!("Deleted {} nodes in {:.2?}", to_delete.len(), elapsed);
    println!(
        "Throughput: {:.0} deletes/sec",
        to_delete.len() as f64 / elapsed.as_secs_f64()
    );
    print_degree_stats("after delete", &index, nrows, &rt);

    let ctx = provider::Context::new();
    for i in 0..nrows as u32 {
        let is_deleted = rt
            .block_on(index.provider().status_by_external_id(&ctx, &i))
            .unwrap()
            .is_deleted();
        if to_delete.contains(&i) {
            assert!(is_deleted, "expected node {} to be deleted", i);
            println!("  Node {} is deleted - OK", i);
        } else {
            assert!(!is_deleted, "expected node {} to NOT be deleted", i);
        }
    }
    println!("Delete verification passed");
}

fn test_dynamic_drop_deleted(data: Arc<diskann_utils::views::Matrix<f32>>) {
    println!("\n=== Dynamic Index: Drop Deleted (consolidate) ===");

    let (index, nrows) = build_index_from_data(data.clone());
    let rt = diskann_benchmark_core::tokio::runtime(2).unwrap();

    build::build(
        SingleInsert::new(
            index.clone(),
            data.clone(),
            provider::Strategy::new(),
            ids::Identity::<u32>::new(),
        ),
        build::Parallelism::dynamic(ONE, ONE),
        &rt,
    )
    .unwrap();

    let ctx = provider::Context::new();
    let to_delete = [0u32, 1, 2, 3, 4];
    for i in to_delete {
        rt.block_on(index.provider().delete(&ctx, &i)).unwrap();
    }
    println!("Marked nodes {:?} as deleted", to_delete);
    print_degree_stats("before consolidate", &index, nrows, &rt);

    let start = Instant::now();
    build::build(
        DropDeleted::new(
            index.clone(),
            false,
            ids::Range::new(0..nrows as u32),
        ),
        build::Parallelism::dynamic(ONE, std::num::NonZeroUsize::new(2).unwrap()),
        &rt,
    )
    .unwrap();
    let elapsed = start.elapsed();

    println!("Consolidation of {} nodes in {:.2?}", nrows, elapsed);
    print_degree_stats("after consolidate", &index, nrows, &rt);

    let accessor = index.provider().neighbors();
    let mut adj = AdjacencyList::new();
    for i in (0..nrows as u32).filter(|i| !to_delete.contains(i)) {
        rt.block_on(accessor.get_neighbors(i, &mut adj)).unwrap();
        for n in adj.iter() {
            assert!(
                !to_delete.contains(n),
                "deleted node {} still appears as neighbor of {}",
                n,
                i
            );
        }
    }
    println!("No deleted nodes appear as neighbors - OK");
}

fn main() {
    let data = load_dataset(DATASET_PATH);
    test_static_index(data.clone());
    test_dynamic_single_insert(data.clone());
    test_dynamic_multi_insert(data.clone());
    test_dynamic_inplace_delete(data.clone());
    test_dynamic_drop_deleted(data.clone());
}
