use std::sync::Arc;
use std::time::Instant;

use diskann::{
    graph::{
        AdjacencyList, DiskANNIndex, InplaceDeleteMethod,
        index::DegreeStats,
        test::{provider, synthetic},
    },
    provider::{Delete, NeighborAccessor},
    utils::{IntoUsize, ONE},
};
use diskann_benchmark_core::{
    build::{
        self,
        graph::{MultiInsert, SingleInsert},
        ids,
    },
    streaming::graph::{DropDeleted, InplaceDelete},
};

fn build_index(
    grid: synthetic::Grid,
    size: usize,
) -> (
    Arc<DiskANNIndex<provider::Provider>>,
    Arc<diskann_utils::views::Matrix<f32>>,
    usize,
) {
    let start_id = u32::MAX;
    let distance = diskann_vector::distance::Metric::L2;

    let start_point = grid.start_point(size);
    let data = Arc::new(grid.data(size));

    let provider_config = provider::Config::new(
        distance,
        2 * grid.dim().into_usize(),
        std::iter::once(provider::StartPoint::new(start_id, start_point)),
    )
    .unwrap();

    let provider = provider::Provider::new(provider_config);

    let index_config = diskann::graph::config::Builder::new(
        provider.max_degree().checked_sub(3).unwrap(),
        diskann::graph::config::MaxDegree::new(provider.max_degree()),
        20,
        distance.into(),
    )
    .build()
    .unwrap();

    let nrows = data.nrows();
    let index = Arc::new(DiskANNIndex::new(index_config, provider, None));

    (index, data, nrows)
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

fn test_static_index() {
    println!("\n=== Static Index ===");

    let grid = synthetic::Grid::Four;
    let size = 4;
    let (index, data, nrows) = build_index(grid, size);
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

fn test_dynamic_single_insert() {
    println!("\n=== Dynamic Index: Single Insert ===");

    let grid = synthetic::Grid::Four;
    let size = 4;
    let (index, data, nrows) = build_index(grid, size);
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

fn test_dynamic_multi_insert() {
    println!("\n=== Dynamic Index: Multi Insert ===");

    let grid = synthetic::Grid::Four;
    let size = 4;
    let (index, data, nrows) = build_index(grid, size);
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

fn test_dynamic_inplace_delete() {
    println!("\n=== Dynamic Index: Inplace Delete ===");

    let grid = synthetic::Grid::Four;
    let size = 4;
    let (index, data, nrows) = build_index(grid, size);
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

    let to_delete: Box<[u32]> = Box::new([0u32, 1, 2]);
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

fn test_dynamic_drop_deleted() {
    println!("\n=== Dynamic Index: Drop Deleted (consolidate) ===");

    let grid = synthetic::Grid::Four;
    let size = 4;
    let (index, data, nrows) = build_index(grid, size);
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
    let to_delete = [0u32, 1];
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
    test_static_index();
    test_dynamic_single_insert();
    test_dynamic_multi_insert();
    test_dynamic_inplace_delete();
    test_dynamic_drop_deleted();
}
