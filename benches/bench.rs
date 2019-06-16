use std::mem;

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, Bencher, Benchmark, Criterion,
};

use hop_slab::HopSlab;
use slab::Slab;

// This function compares benchmark tests for the 4 main code paths that can occur when inserting
// entries:
//
// - hop_slab_insert_no_vacant_entries: there are no vacant entries, the new entry is simply
//   appended to the backing Vec.
// - hop_slab_insert_shrink_free_block: there are vacant entries and inserting the new entry will
//   shrink a free block but not eliminate it.
// - hop_slab_insert_unlink_free_block: there are vacant entries and inserting the new entry will
//   eliminate a free block and unlink it.
//
// For reference to also contains benchmark tests for the 2 main code paths for slab::Slab::insert:
//
// - slab_insert_no_vacant_entries: there are no vacant entries, the new entry is simply appended to
//   the backing Vec.
// - slab_insert_with_vacant_entries: there are vacant entries, the new entry will eliminate and
//   unlink a vacant entry.
fn insert(criterion: &mut Criterion) {
    fn hop_slab_insert_no_vacant_entries(bencher: &mut Bencher) {
        bencher.iter_batched_ref(
            || HopSlab::with_capacity(1),
            |slab| slab.insert("a"),
            BatchSize::LargeInput,
        )
    }

    fn hop_slab_insert_shrink_free_block(bencher: &mut Bencher) {
        bencher.iter_batched_ref(
            || {
                let mut slab = HopSlab::with_capacity(3);

                let a = slab.insert("a");
                let b = slab.insert("b");
                slab.insert("c");
                slab.remove(a);
                slab.remove(b);

                slab
            },
            |slab| slab.insert("d"),
            BatchSize::LargeInput,
        )
    }

    fn hop_slab_insert_unlink_free_block(bencher: &mut Bencher) {
        bencher.iter_batched_ref(
            || {
                let mut slab = HopSlab::with_capacity(3);

                let a = slab.insert("a");
                slab.insert("b");
                slab.insert("c");
                slab.remove(a);

                slab
            },
            |slab| slab.insert("c"),
            BatchSize::LargeInput,
        )
    }

    fn slab_insert_no_vacant_entries(bencher: &mut Bencher) {
        bencher.iter_batched_ref(
            || Slab::with_capacity(1),
            |slab| slab.insert("a"),
            BatchSize::LargeInput,
        )
    }

    fn slab_insert_with_vacant_entries(bencher: &mut Bencher) {
        bencher.iter_batched_ref(
            || {
                let mut slab = Slab::with_capacity(3);

                let a = slab.insert("a");
                let b = slab.insert("b");
                slab.insert("c");
                slab.remove(a);
                slab.remove(b);

                slab
            },
            |slab| slab.insert("d"),
            BatchSize::LargeInput,
        )
    }

    let benchmark = Benchmark::new(
        "HopSlab::insert no vacant entries",
        hop_slab_insert_no_vacant_entries,
    )
    .with_function(
        "HopSlab::insert shrink free block",
        hop_slab_insert_shrink_free_block,
    )
    .with_function(
        "HopSlab::insert unlink free block",
        hop_slab_insert_unlink_free_block,
    )
    .with_function(
        "Slab::insert no vacant entries",
        slab_insert_no_vacant_entries,
    )
    .with_function(
        "Slab::insert with vacant entries",
        slab_insert_with_vacant_entries,
    );

    criterion.bench("insert", benchmark);
}

// This functions compares benchmark tests for the 5 main code path that may occur when removing an
// entry from a HopSlab:
//
// - hop_slab_remove_no_adjacent_free_blocks: the entry that is being removed is not adjacent to
//   any vacant entries.
// - hop_slab_remove_preceding_free_block: the entry that is being removed is preceded by a block
//   of vacant entries.
// - hop_slab_remove_succeeding_free_block: the entry that is being removed is succeeded by a block
//   of vacant entries.
// - hop_slab_remove_preceding_and_succeeding_free_block: the entry that is being removed is both
//   preceded by a block of vacant entries and succeeded by a block of vacant entries.
// - hop_slab_remove_last_entry: the entry that is being removed is the very last entry that is
//   currently stored.
//
// All of these paths have different performance characteristics. Note that these paths are not
// equally likely to occur and that the rate on occurrence depends on the current fill-rate of the
// HopSlab: in a very densely filled HopMap (few vacant entries)
// hop_slab_remove_no_adjacent_free_blocks is relatively likely, whereas in a very sparely filled
// HopSlab, hop_slab_remove_preceding_and_succeeding_free_block is relatively likely.
//
// For reference this comparison also contains a benchmark test for slab::Slab::remove.
fn remove(criterion: &mut Criterion) {
    fn hop_slab_remove_no_adjacent_free_blocks(bencher: &mut Bencher) {
        bencher.iter_batched_ref(
            || {
                let mut slab = HopSlab::new();

                slab.insert("a");
                let b = slab.insert("b");
                slab.insert("c");

                (slab, b)
            },
            |(slab, key)| slab.remove(*key),
            BatchSize::LargeInput,
        )
    }

    fn hop_slab_remove_preceding_free_block(bencher: &mut Bencher) {
        bencher.iter_batched_ref(
            || {
                let mut slab = HopSlab::new();

                let a = slab.insert("a");
                let b = slab.insert("b");
                slab.insert("c");

                slab.remove(a);

                (slab, b)
            },
            |(slab, key)| slab.remove(*key),
            BatchSize::LargeInput,
        )
    }

    fn hop_slab_remove_succeeding_free_block(bencher: &mut Bencher) {
        bencher.iter_batched_ref(
            || {
                let mut slab = HopSlab::new();

                slab.insert("a");
                let b = slab.insert("b");
                let c = slab.insert("c");
                slab.insert("d");

                slab.remove(c);

                (slab, b)
            },
            |(slab, key)| slab.remove(*key),
            BatchSize::LargeInput,
        )
    }

    fn hop_slab_remove_preceding_and_succeeding_free_block(bencher: &mut Bencher) {
        bencher.iter_batched_ref(
            || {
                let mut slab = HopSlab::new();

                let a = slab.insert("a");
                let b = slab.insert("b");
                let c = slab.insert("c");
                slab.insert("d");

                slab.remove(a);
                slab.remove(c);

                (slab, b)
            },
            |(slab, key)| slab.remove(*key),
            BatchSize::LargeInput,
        )
    }

    fn hop_slab_remove_last_entry(bencher: &mut Bencher) {
        bencher.iter_batched_ref(
            || {
                let mut slab = HopSlab::new();

                slab.insert("a");
                let b = slab.insert("b");

                (slab, b)
            },
            |(slab, key)| slab.remove(*key),
            BatchSize::LargeInput,
        )
    }

    fn slab_remove(bencher: &mut Bencher) {
        bencher.iter_batched_ref(
            || {
                let mut slab = Slab::new();

                slab.insert("a");
                let b = slab.insert("b");
                slab.insert("c");

                (slab, b)
            },
            |(slab, key)| slab.remove(*key),
            BatchSize::LargeInput,
        )
    }

    let benchmark = Benchmark::new(
        "HopSlab::remove no adjacent free blocks",
        hop_slab_remove_no_adjacent_free_blocks,
    )
    .with_function(
        "HopSlab::remove preceding free block",
        hop_slab_remove_preceding_free_block,
    )
    .with_function(
        "HopSlab::remove succeeding free block",
        hop_slab_remove_succeeding_free_block,
    )
    .with_function(
        "HopSlab::remove preceding and succeeding free block",
        hop_slab_remove_preceding_and_succeeding_free_block,
    )
    .with_function("HopSlab::remove last entry", hop_slab_remove_last_entry)
    .with_function("Slab::remove", slab_remove);

    criterion.bench("remove", benchmark);
}

fn iter(criterion: &mut Criterion) {
    fn hop_slab_iter_dense(bencher: &mut Bencher) {
        bencher.iter_batched_ref(
            || {
                let mut slab = HopSlab::with_capacity(1000);

                for i in 0..1000 {
                    slab.insert(i);
                }

                for key in 1..1001 {
                    if key % 50 == 0 {
                        slab.remove(key);
                    }
                }

                slab
            },
            |slab| {
                for entry in slab.iter() {
                    black_box(entry);
                }
            },
            BatchSize::LargeInput,
        )
    }

    fn hop_slab_iter_sparse(bencher: &mut Bencher) {
        bencher.iter_batched_ref(
            || {
                let mut slab = HopSlab::with_capacity(1000);

                for i in 0..1000 {
                    slab.insert(i);
                }

                for key in 1..1001 {
                    if key % 50 != 0 {
                        slab.remove(key);
                    }
                }

                slab
            },
            |slab| {
                for entry in slab.iter() {
                    black_box(entry);
                }
            },
            BatchSize::LargeInput,
        )
    }

    fn slab_iter_dense(bencher: &mut Bencher) {
        bencher.iter_batched_ref(
            || {
                let mut slab = Slab::with_capacity(1000);

                for i in 0..1000 {
                    slab.insert(i);
                }

                for key in 0..1000 {
                    if key % 50 == 0 {
                        slab.remove(key);
                    }
                }

                slab
            },
            |slab| {
                for entry in slab.iter() {
                    black_box(entry);
                }
            },
            BatchSize::LargeInput,
        )
    }

    fn slab_iter_sparse(bencher: &mut Bencher) {
        bencher.iter_batched_ref(
            || {
                let mut slab = Slab::with_capacity(1000);

                for i in 0..1000 {
                    slab.insert(i);
                }

                for key in 0..1000 {
                    if key % 50 != 0 {
                        slab.remove(key);
                    }
                }

                slab
            },
            |slab| {
                for entry in slab.iter() {
                    black_box(entry);
                }
            },
            BatchSize::LargeInput,
        )
    }

    let benchmark = Benchmark::new("HopSlab::iter dense occupation rate", hop_slab_iter_dense)
        .with_function("HopSlab::iter sparse occupation rate", hop_slab_iter_sparse)
        .with_function("Slab::iter dense occupation rate", slab_iter_dense)
        .with_function("Slab::iter sparse occupation rate", slab_iter_sparse);

    criterion.bench("iter", benchmark);
}

criterion_group!(benches, insert, remove, iter);
criterion_main!(benches);
