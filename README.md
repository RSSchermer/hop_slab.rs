# HopSlab.rs

Pre-allocated storage for a uniform data type, in which vacant slots are tracked in blocks
rather than individually.

The `HopSlab` data structure exposes an interface that is very similar to the interface of the
`Slab` data structure as implemented by the [slab](https://crates.io/crates/slab). However, the
`HopSlab` implementation tracks the empty slots in its memory differently, which results in a
different performance trade-off: iterating over a sparsely occupied `HopSlab` may be much
quicker than iterating over a sparsely occupied `Slab`, as the `HopSlab` may "hop" over blocks
of consecutive vacant slots in a single bound, where a `Slab` will inspect every vacant slot.
More specifically, the cost of iterating over a `Slab` is determined primarily by the occupied
slot with the largest key and only secondarily by the total number of occupied slots, whereas
the cost of iterating over a `HopSlab` is determined primarily by the total number of occupied
slots and only secondarily by the occupied slot with the largest key. Phrased differently: if
the occupation rate of the slab may vary greatly (such that slab occupation may at times be
very sparse), then the cost of iterating over a `HopSlab` may be more predictable than the cost
of iterating over a `Slab`.

However, this requires more complex book-keeping when inserting and removing entries and thus
these operations become more expensive; as a rough guide, inserting is up to 1.5 times slower
and removing is up to 2 times slower. There is some nuance to this, see [benches/bench.rs](benches/bench.rs) 
for details. The cost of lookups is identical for both slab implementations.

Additionally, a `HopSlab` may use more memory than a regular `Slab`. Both the `HopSlab` and the
`Slab` use vacant slots themselves to track the location of the vacant slots. However, a `Slab`
requires only a single `usize` value to do so, whereas a `HopSlab` uses 3 `usize` values. If the
size of the data type you are storing in the slab is equal to or smaller than the size of
`usize`, a `HopSlab` may require nearly 3 times as much memory as a `Slab` of the same
capacity. This difference does not apply when storing larger data types: if the size of the data
type is equal to or greater than `3 * std::mem::size_of::<usize>()`, then the memory usage of a
`HopSlab` should be nearly identical to that of a `Slab` of the same capacity (the `HopSlab`
will allocate one additional sentinel entry).

The majority of use-cases for a slab-like storage structures will likely benefit more from the
regular `Slab` implementation. Only if the performance of iterating over the entries in the slab
dominates your use-case, consider swapping it out for a `HopSlab` and benchmark the difference.