#![feature(ptr_offset_from)]

//! Pre-allocated storage for a uniform data type, in which vacant slots are tracked in blocks
//! rather than individually.
//!
//! The [HopSlab] data structure exposes an interface that is very similar to the interface of the
//! `Slab` data structure as implemented by the [slab](https://crates.io/crates/slab). However, the
//! [HopSlab] implementation tracks the empty slots in its memory differently, which results in a
//! different performance trade-off: iterating over a sparsely occupied [HopSlab] may be much
//! quicker than iterating over a sparsely occupied `Slab`, as the [HopSlab] may "hop" over blocks
//! of consecutive vacant slots in a single bound, where a `Slab` will inspect every vacant slot.
//! More specifically, the cost of iterating over a `Slab` is determined primarily by the occupied
//! slot with the largest key and only secondarily by the total number of occupied slots, whereas
//! the cost of iterating over a [HopSlab] is determined primarily by the total number of occupied
//! slots and only secondarily by the occupied slot with the largest key. Phrased differently: if
//! the occupation rate of the slab may vary greatly (such that slab occupation may at times be
//! very sparse), then the cost of iterating over a [HopSlab] may be more predictable than the cost
//! of iterating over a `Slab`.
//!
//! However, this requires more complex book-keeping when inserting and removing entries and thus
//! these operations become more expensive; as a rough guide, inserting is up to 1.5 times slower
//! and removing is up to 2 times slower. There is some nuance to this, see [benches/bench.rs] for
//! details. The cost of lookups is identical for both slab implementations.
//!
//! Additionally, a [HopSlab] may use more memory than a regular `Slab`. Both the [HopSlab] and the
//! `Slab` use vacant slots themselves to track the location of the vacant slots. However, a `Slab`
//! requires only a single `usize` value to do so, whereas a [HopSlab] uses 3 `usize` values. If the
//! size of the data type you are storing in the slab is equal to or smaller than the size of
//! `usize`, a [HopSlab] may require nearly 3 times as much memory as a `Slab` of the same
//! capacity. This difference does not apply when storing larger data types: if the size of the data
//! type is equal to or greater than `3 * std::mem::size_of::<usize>()`, then the memory usage of a
//! [HopSlab] should be nearly identical to that of a `Slab` of the same capacity (the [HopSlab]
//! will allocate one additional sentinel entry).
//!
//! The majority of use-cases for a slab-like storage structures will likely benefit more from the
//! regular `Slab` implementation. Only if the performance of iterating over the entries in the slab
//! dominates your use-case, consider swapping it out for a [HopSlab] and benchmark the difference.

use std::{marker, mem};
use std::hint::unreachable_unchecked;
use std::num::NonZeroUsize;
use std::ops::{Index, IndexMut};

#[derive(PartialEq, Debug)]
enum Entry<T> {
    Occupied(T),
    FreeBlockHead(FreeBlockHead),
    FreeBlockTail(usize),
}

impl<T> Entry<T> {
    fn is_occupied(&self) -> bool {
        if let Entry::Occupied(_) = self {
            true
        } else {
            false
        }
    }

    fn get(&self) -> Option<&T> {
        if let Entry::Occupied(value) = self {
            Some(value)
        } else {
            None
        }
    }

    fn get_mut(&mut self) -> Option<&mut T> {
        if let Entry::Occupied(value) = self {
            Some(value)
        } else {
            None
        }
    }
}

impl<T> From<FreeBlockHead> for Entry<T> {
    fn from(block: FreeBlockHead) -> Self {
        Entry::FreeBlockHead(block)
    }
}

#[derive(PartialEq, Debug)]
struct FreeBlockHead {
    previous: Option<NonZeroUsize>,
    next: Option<NonZeroUsize>,
    end: usize,
}

pub struct VacantEntry<'a, T> {
    slab: &'a mut HopSlab<T>,
}

impl<'a, T> VacantEntry<'a, T> {
    pub fn key(&self) -> usize {
        if let Some(index) = self.slab.first_free_block {
            let block = unsafe { self.slab.entries.get_unchecked(index.get()) };

            if let Entry::FreeBlockHead(block) = block {
                block.end - 1
            } else {
                unsafe { unreachable_unchecked() }
            }
        } else {
            self.slab.entries.len()
        }
    }

    pub fn insert(self, value: T) -> &'a mut T {
        let key = self.slab.insert(value);

        unsafe {
            if let Entry::Occupied(value) = self.slab.entries.get_unchecked_mut(key) {
                value
            } else {
                unsafe { unreachable_unchecked() }
            }
        }
    }
}

pub struct HopSlab<T> {
    entries: Vec<Entry<T>>,
    len: usize,
    first_free_block: Option<NonZeroUsize>,
}

impl<T> HopSlab<T> {
    pub fn new() -> Self {
        HopSlab {
            entries: vec![Entry::FreeBlockTail(0)],
            len: 0,
            first_free_block: None,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == std::usize::MAX {
            panic!("Max capacity exceeded.")
        }

        let mut entries = Vec::with_capacity(capacity + 1);

        entries.push(Entry::FreeBlockTail(0));

        HopSlab {
            entries,
            len: 0,
            first_free_block: None,
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.entries.push(Entry::FreeBlockTail(0));
        self.len = 0;
        self.first_free_block = None;
    }

    pub fn capacity(&self) -> usize {
        self.entries.capacity() - 1
    }

    pub fn reserve(&mut self, additional: usize) {
        if self.capacity() - self.len() < additional {
            let additional_entries = self.len() + 1 + additional - self.entries.len();

            self.entries.reserve(additional_entries);
        }
    }

    pub fn reserve_exact(&mut self, additional: usize) {
        if self.capacity() - self.len() < additional {
            let additional_entries = self.len() + 1 + additional - self.entries.len();

            self.entries.reserve_exact(additional_entries);
        }
    }

    pub fn shrink_to_fit(&mut self) {
        self.entries.shrink_to_fit();
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn contains(&self, key: usize) -> bool {
        self.entries
            .get(key)
            .map(|entry| entry.is_occupied())
            .unwrap_or(false)
    }

    pub fn get(&self, key: usize) -> Option<&T> {
        self.entries.get(key).and_then(|entry| entry.get())
    }

    pub unsafe fn get_unchecked(&self, key: usize) -> &T {
        if let Entry::Occupied(value) = self.entries.get_unchecked(key) {
            value
        } else {
            unsafe { unreachable_unchecked() }
        }
    }

    pub fn get_mut(&mut self, key: usize) -> Option<&mut T> {
        self.entries.get_mut(key).and_then(|entry| entry.get_mut())
    }

    pub unsafe fn get_unchecked_mut(&mut self, key: usize) -> &T {
        if let Entry::Occupied(value) = self.entries.get_unchecked_mut(key) {
            value
        } else {
            unsafe { unreachable_unchecked() }
        }
    }

    pub fn insert(&mut self, value: T) -> usize {
        self.len += 1;

        if let Some(free_block_index) = self.first_free_block {
            let free_block_index = free_block_index.get();
            let entry = unsafe { self.entries.get_unchecked_mut(free_block_index) };

            if let Entry::FreeBlockHead(block) = entry {
                if block.end == free_block_index + 1 {
                    // We're in a free block that consists only of a head entry, which means this
                    // free block will be eliminated. We need to unlink it and simply replace it
                    // with the new data entry.

                    let next = block.next;

                    *entry = Entry::Occupied(value);

                    self.first_free_block = next;

                    next.map(|index| {
                        let entry = unsafe { self.entries.get_unchecked_mut(index.get()) };

                        if let Entry::FreeBlockHead(block) = entry {
                            block.previous = None;
                        } else {
                            unsafe { unreachable_unchecked() };
                        }
                    });

                    free_block_index
                } else if block.end == free_block_index + 2 {
                    // We're in a block that consists of a head entry and exactly 1 tail entry.
                    // Simply replacing the tail entry will suffice.

                    block.end -= 1;

                    let index = free_block_index + 1;
                    let entry = unsafe { self.entries.get_unchecked_mut(index) };

                    *entry = Entry::Occupied(value);

                    index
                } else {
                    // We're in a block that consists of a head entry and more than 1 tail entries.
                    // We'll replace the last tail entry with the new data entry, but we must also
                    // ensure that the second to last entry (the entry that becomes the new tail of
                    // the block) correctly points to the start of the block, as it is not
                    // guaranteed to do so if it resulted from a block merger (see also [remove]).

                    block.end -= 1;

                    let index = block.end;

                    let new_tail = unsafe { self.entries.get_unchecked_mut(index - 1) };

                    if let Entry::FreeBlockTail(start) = new_tail {
                        *start = free_block_index;
                    } else {
                        unsafe { unreachable_unchecked() };
                    }

                    let entry = unsafe { self.entries.get_unchecked_mut(index) };

                    *entry = Entry::Occupied(value);

                    index
                }
            } else {
                unsafe { unreachable_unchecked() };
            }
        } else {
            let index = self.entries.len();

            self.entries.push(Entry::Occupied(value));

            index
        }
    }

    pub fn remove(&mut self, key: usize) -> Option<T> {
        if let Some(entry) = self.entries.get_mut(key) {
            if entry.is_occupied() {
                let entry = entry as *mut _;

                self.len -= 1;

                // Special case for when key points to the last entry. This is both a fast case and
                // also allows us to assume that the key is not the last key in the alternative
                // case.
                if key == self.entries.len() - 1 {
                    unsafe {
                        let entry = mem::replace(&mut *entry, Entry::FreeBlockTail(0));

                        // Note that we know that key is not zero, as the 0 position contains an
                        // unoccupied entry and thus the is_occupied check above would have failed.
                        self.entries.set_len(key);

                        if let Entry::Occupied(value) = entry {
                            return Some(value);
                        } else {
                            unsafe { unreachable_unchecked() }
                        }
                    }
                }

                // Safe because there's always an unoccupied sentry entry at index 0
                let preceding_entry = unsafe { self.entries.get_unchecked_mut(key - 1) };

                let (preceding_block, preceding_block_index) = match preceding_entry {
                    Entry::Occupied(_) => (None, 0),
                    Entry::FreeBlockHead(block) => (Some(block as *mut FreeBlockHead), key - 1),
                    Entry::FreeBlockTail(head_index) => {
                        let head_index = *head_index;
                        let head = unsafe { self.entries.get_unchecked_mut(head_index) };

                        if let Entry::FreeBlockHead(block) = head {
                            (Some(block as *mut FreeBlockHead), head_index)
                        } else {
                            // Note that the only case in which a block tail does not point to a
                            // block head may be the case in which the block tail is the sentinel
                            // entry at position 0, in which case the current key must be 1.
                            debug_assert_eq!(key, 1);

                            (None, 0)
                        }
                    }
                };

                // Safe because we've already checked whether or not the key is the last index.
                let mut succeeding_entry = unsafe { self.entries.get_unchecked_mut(key + 1) };

                let entry = match (preceding_block, &mut succeeding_entry) {
                    (None, Entry::Occupied(_)) => {
                        // The entry that is being removed is not adjacent to any blocks of vacant
                        // entries. Replace it with a new free block and prepend the block to the
                        // free block chain.

                        let first_free_block = self.first_free_block;

                        if let Some(index) = first_free_block {
                            let entry = unsafe { self.entries.get_unchecked_mut(index.get()) };

                            if let Entry::FreeBlockHead(block) = entry {
                                block.previous = unsafe { Some(NonZeroUsize::new_unchecked(key)) };
                            } else {
                                unsafe { unreachable_unchecked() };
                            }
                        }

                        self.first_free_block = unsafe { Some(NonZeroUsize::new_unchecked(key)) };

                        let block = FreeBlockHead {
                            previous: None,
                            next: first_free_block,
                            end: key + 1,
                        };

                        unsafe { mem::replace(&mut *entry, block.into()) }
                    }
                    (Some(block), Entry::Occupied(_)) => {
                        // The entry that is being removed is preceded by a block of vacant entries.
                        // Increment the size of that block and replace the entry with a new free
                        // block tail entry.

                        unsafe {
                            (*block).end += 1;

                            mem::replace(&mut *entry, Entry::FreeBlockTail(preceding_block_index))
                        }
                    }
                    (None, Entry::FreeBlockHead(block)) => {
                        // The entry that is being removed is succeeded by a block of vacant
                        // entries. Move the head of that block into the position of the entry that
                        // is being removed and fill the "hole" with a new tail entry.

                        let end = block.end;
                        let block_head = mem::replace(succeeding_entry, Entry::FreeBlockTail(key));

                        // If the block has a tail entry, update it to point to the new head
                        // position.
                        if end > key + 2 {
                            let tail = unsafe { self.entries.get_unchecked_mut(end - 1) };

                            *tail = Entry::FreeBlockTail(key);
                        }

                        unsafe { mem::replace(&mut *entry, block_head) }
                    }
                    (Some(preceding_block), Entry::FreeBlockHead(succeeding_block)) => {
                        // The entry that is being removed is both preceded and succeeded by blocks
                        // of vacant entries. Merge the succeeding block into the preceding block
                        // and unlink the succeeding block from the free block chain.

                        let succeeding_block_previous = succeeding_block.previous;
                        let succeeding_block_next = succeeding_block.next;
                        let succeeding_block_end = succeeding_block.end;

                        *succeeding_entry = Entry::FreeBlockTail(preceding_block_index);

                        // Unlink the next block from the free block chain
                        if let Some(succeeding_block_previous) = succeeding_block_previous {
                            let entry = unsafe {
                                self.entries
                                    .get_unchecked_mut(succeeding_block_previous.get())
                            };

                            if let Entry::FreeBlockHead(block) = entry {
                                block.next = succeeding_block_next;
                            } else {
                                unsafe { unreachable_unchecked() };
                            }
                        } else {
                            self.first_free_block = succeeding_block_next;
                        }

                        if let Some(succeeding_block_next) = succeeding_block_next {
                            let entry = unsafe {
                                self.entries.get_unchecked_mut(succeeding_block_next.get())
                            };

                            if let Entry::FreeBlockHead(block) = entry {
                                block.previous = succeeding_block_previous;
                            } else {
                                unsafe { unreachable_unchecked() };
                            }
                        }

                        // If the succeeding block has a tail entry, then update that tail entry to
                        // point to the head of the preceding block.
                        if succeeding_block_end > key + 2 {
                            let tail =
                                unsafe { self.entries.get_unchecked_mut(succeeding_block_end - 1) };

                            *tail = Entry::FreeBlockTail(preceding_block_index);
                        }

                        unsafe {
                            // Finally, update the size of the preceding block
                            (*preceding_block).end = succeeding_block_end;

                            mem::replace(&mut *entry, Entry::FreeBlockTail(preceding_block_index))
                        }
                    }
                    _ => unsafe { unreachable_unchecked() },
                };

                if let Entry::Occupied(value) = entry {
                    Some(value)
                } else {
                    unsafe { unreachable_unchecked() }
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn vacant_entry(&mut self) -> VacantEntry<T> {
        VacantEntry { slab: self }
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &mut T) -> bool,
    {
        let mut i = 1;

        while i < self.entries.len() {
            let key = i - 1;

            if let Entry::FreeBlockHead(block) = &self.entries[i] {
                i = block.end;
            } else {
                i += 1;
            }

            if let Entry::Occupied(value) = &mut self.entries[key] {
                if !f(key, value) {
                    self.remove(key);
                }
            }
        }

        if let Entry::Occupied(value) = &mut self.entries[i - 1] {
            if !f(i - 1, value) {
                self.remove(i - 1);
            }
        }
    }

    pub fn drain(&mut self) -> Drain<T> {
        let origin = self.entries.as_mut_ptr();
        let entry_count = self.entries.len();

        unsafe {
            Drain {
                remaining: self.len,
                slab: self,
                current_front: origin.offset(1),
                current_back: origin.offset(entry_count as isize - 1),
                origin,
            }
        }
    }

    pub fn compact<F>(&mut self, mut move_fn: F)
    where
        F: FnMut(&mut T, usize, usize) -> bool,
    {
        struct CleanupGuard<'a, T> {
            slab: &'a mut HopSlab<T>,
            write_cursor: usize,
            read_cursor: usize,
        }

        impl<'a, T> CleanupGuard<'a, T> {
            fn cleanup(&mut self) {
                // Stop compacting. First fill the current gap between te read and write cursors
                // with a new free block and then iterate through the remaining entries to recreate
                // the free block chain.

                debug_assert!(self.write_cursor < self.read_cursor);

                let CleanupGuard {
                    slab,
                    write_cursor,
                    read_cursor,
                } = self;

                let mut last_block;

                unsafe {
                    *slab.entries.get_unchecked_mut(*write_cursor) = FreeBlockHead {
                        previous: None,
                        next: None,
                        end: *read_cursor,
                    }
                    .into();

                    slab.first_free_block = Some(NonZeroUsize::new_unchecked(*write_cursor));

                    *slab.entries.get_unchecked_mut(*read_cursor - 1) =
                        Entry::FreeBlockTail(*write_cursor);

                    last_block = *write_cursor;
                }

                *read_cursor += 1;

                while *read_cursor < slab.entries.len() {
                    let entry = unsafe { slab.entries.get_unchecked_mut(*read_cursor) };

                    if let Entry::FreeBlockHead(block) = entry {
                        unsafe {
                            let end = block.end;

                            block.previous = None;
                            block.next = Some(NonZeroUsize::new_unchecked(last_block));

                            let block = slab.entries.get_unchecked_mut(last_block);

                            if let Entry::FreeBlockHead(block) = block {
                                block.previous = Some(NonZeroUsize::new_unchecked(*read_cursor));
                            } else {
                                unsafe { unreachable_unchecked() }
                            }

                            slab.first_free_block = Some(NonZeroUsize::new_unchecked(*read_cursor));
                            last_block = *read_cursor;

                            *read_cursor = end;
                        }
                    } else {
                        *read_cursor += 1;
                    }
                }
            }
        }

        impl<'a, T: 'a> Drop for CleanupGuard<'a, T> {
            fn drop(&mut self) {
                if self.read_cursor < self.slab.entries.len() {
                    self.cleanup();
                }
            }
        }

        if self.first_free_block.is_some() {
            let mut guard = CleanupGuard {
                slab: self,
                write_cursor: 1,
                read_cursor: 0,
            };

            loop {
                match unsafe { guard.slab.entries.get_unchecked(guard.write_cursor) } {
                    Entry::FreeBlockHead(block) => {
                        guard.read_cursor = block.end;

                        break;
                    }
                    _ => {
                        guard.write_cursor += 1;
                    }
                }
            }

            while guard.read_cursor < guard.slab.entries.len() {
                let entry = unsafe { guard.slab.entries.get_unchecked_mut(guard.read_cursor) };

                match entry {
                    Entry::FreeBlockHead(block) => {
                        guard.read_cursor = block.end;
                    }
                    Entry::Occupied(element) => {
                        if move_fn(element, guard.read_cursor, guard.write_cursor) {
                            let entry = mem::replace(entry, Entry::FreeBlockTail(0));

                            unsafe {
                                *guard.slab.entries.get_unchecked_mut(guard.write_cursor) = entry;
                            }

                            guard.read_cursor += 1;
                            guard.write_cursor += 1;
                        } else {
                            guard.cleanup();

                            return;
                        }
                    }
                    _ => unsafe { unreachable_unchecked() },
                }
            }

            unsafe { guard.slab.entries.set_len(guard.write_cursor) };

            guard.slab.first_free_block = None;
        }
    }

    pub fn iter(&self) -> Iter<T> {
        let origin = self.entries.as_ptr();
        let entry_count = self.entries.len();

        unsafe {
            Iter {
                current_front: origin.offset(1),
                current_back: origin.offset(entry_count as isize - 1),
                origin,
                remaining: self.len,
                _marker: marker::PhantomData,
            }
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<T> {
        let origin = self.entries.as_mut_ptr();
        let entry_count = self.entries.len();

        unsafe {
            IterMut {
                current_front: origin.offset(1),
                current_back: origin.offset(entry_count as isize - 1),
                origin,
                remaining: self.len,
                _marker: marker::PhantomData,
            }
        }
    }
}

impl<T> Default for HopSlab<T> {
    fn default() -> Self {
        HopSlab::new()
    }
}

impl<T> Index<usize> for HopSlab<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match self.entries.get(index) {
            Some(Entry::Occupied(value)) => value,
            _ => panic!("invalid key"),
        }
    }
}

impl<T> IndexMut<usize> for HopSlab<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self.entries.get_mut(index) {
            Some(Entry::Occupied(value)) => value,
            _ => panic!("invalid key"),
        }
    }
}

impl<T> IntoIterator for HopSlab<T> {
    type Item = (usize, T);

    type IntoIter = IntoIter<T>;

    fn into_iter(mut self) -> Self::IntoIter {
        let origin = self.entries.as_mut_ptr();
        let entry_count = self.entries.len();

        unsafe {
            IntoIter {
                current_front: origin.offset(1),
                current_back: origin.offset(entry_count as isize - 1),
                origin,
                remaining: self.len,
                slab: self,
            }
        }
    }
}

impl<'a, T> IntoIterator for &'a HopSlab<T> {
    type Item = (usize, &'a T);

    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut HopSlab<T> {
    type Item = (usize, &'a mut T);

    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

pub struct IntoIter<T> {
    slab: HopSlab<T>,
    origin: *mut Entry<T>,
    current_front: *mut Entry<T>,
    current_back: *mut Entry<T>,
    remaining: usize,
}

impl<T> Iterator for IntoIter<T> {
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining > 0 {
            self.remaining -= 1;

            unsafe {
                let entry = mem::replace(&mut *self.current_front, Entry::FreeBlockTail(0));

                match entry {
                    Entry::Occupied(value) => {
                        let key = self.current_front.offset_from(self.origin) as usize;

                        self.current_front = self.current_front.offset(1);

                        Some((key, value))
                    }
                    Entry::FreeBlockHead(block) => {
                        let end = block.end;

                        let entry = self.origin.offset(end as isize);

                        self.current_front = entry.offset(1);

                        let entry = mem::replace(&mut *entry, Entry::FreeBlockTail(0));

                        if let Entry::Occupied(value) = entry {
                            Some((end, value))
                        } else {
                            unsafe { unreachable_unchecked() }
                        }
                    }
                    _ => unsafe { unreachable_unchecked() },
                }
            }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }

    fn count(self) -> usize {
        self.len()
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining > 0 {
            self.remaining -= 1;

            unsafe {
                let entry = mem::replace(&mut *self.current_back, Entry::FreeBlockTail(0));

                match entry {
                    Entry::Occupied(value) => {
                        let key = self.current_back.offset_from(self.origin) as usize;

                        self.current_back = self.current_back.offset(-1);

                        Some((key, value))
                    }
                    Entry::FreeBlockHead(_) => {
                        let entry = self.current_back.offset(-1);
                        let key = entry.offset_from(self.origin) as usize;

                        self.current_back = entry.offset(-1);

                        let entry = mem::replace(&mut *entry, Entry::FreeBlockTail(0));

                        if let Entry::Occupied(value) = entry {
                            Some((key, value))
                        } else {
                            unsafe { unreachable_unchecked() }
                        }
                    }
                    Entry::FreeBlockTail(head_index) => {
                        let end = head_index - 1;
                        let entry = self.origin.offset(end as isize);

                        self.current_back = entry.offset(-1);

                        let entry = mem::replace(&mut *entry, Entry::FreeBlockTail(0));

                        if let Entry::Occupied(value) = entry {
                            Some((end, value))
                        } else {
                            unsafe { unreachable_unchecked() }
                        }
                    },
                }
            }
        } else {
            None
        }
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {
    fn len(&self) -> usize {
        self.remaining
    }
}

pub struct Iter<'a, T> {
    origin: *const Entry<T>,
    current_front: *const Entry<T>,
    current_back: *const Entry<T>,
    remaining: usize,
    _marker: marker::PhantomData<&'a HopSlab<T>>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining > 0 {
            self.remaining -= 1;

            unsafe {
                match &*self.current_front {
                    Entry::Occupied(value) => {
                        let key = self.current_front.offset_from(self.origin) as usize;

                        self.current_front = self.current_front.offset(1);

                        Some((key, value))
                    }
                    Entry::FreeBlockHead(block) => {
                        let end = block.end;
                        let entry = self.origin.offset(end as isize);

                        self.current_front = entry.offset(1);

                        if let Entry::Occupied(value) = &*entry {
                            Some((end, value))
                        } else {
                            unsafe { unreachable_unchecked() }
                        }
                    }
                    _ => unsafe { unreachable_unchecked() },
                }
            }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }

    fn count(self) -> usize {
        self.len()
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining > 0 {
            self.remaining -= 1;

            unsafe {
                match &*self.current_back {
                    Entry::Occupied(value) => {
                        let key = self.current_back.offset_from(self.origin) as usize;

                        self.current_back = self.current_back.offset(-1);

                        Some((key, value))
                    },
                    Entry::FreeBlockHead(_) => {
                        let entry = self.current_back.offset(-1);
                        let key = entry.offset_from(self.origin) as usize;

                        self.current_back = entry.offset(-1);

                        if let Entry::Occupied(value) = &*entry {
                            Some((key, value))
                        } else {
                            unsafe { unreachable_unchecked() }
                        }
                    },
                    Entry::FreeBlockTail(head_index) => {
                        let end = head_index - 1;
                        let entry = self.origin.offset(end as isize);

                        self.current_back = entry.offset(-1);

                        if let Entry::Occupied(value) = &*entry {
                            Some((end, value))
                        } else {
                            unsafe { unreachable_unchecked() }
                        }
                    }
                }
            }
        } else {
            None
        }
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {
    fn len(&self) -> usize {
        self.remaining
    }
}

pub struct IterMut<'a, T> {
    origin: *mut Entry<T>,
    current_front: *mut Entry<T>,
    current_back: *mut Entry<T>,
    remaining: usize,
    _marker: marker::PhantomData<&'a HopSlab<T>>,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = (usize, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining > 0 {
            self.remaining -= 1;

            unsafe {
                match &mut *self.current_front {
                    Entry::Occupied(value) => {
                        let key = self.current_front.offset_from(self.origin) as usize;

                        self.current_front = self.current_front.offset(1);

                        Some((key, value))
                    }
                    Entry::FreeBlockHead(block) => {
                        let end = block.end;
                        let entry = self.origin.offset(end as isize);

                        self.current_front = entry.offset(1);

                        if let Entry::Occupied(value) = &mut *entry {
                            Some((end, value))
                        } else {
                            unsafe { unreachable_unchecked() }
                        }
                    }
                    _ => unsafe { unreachable_unchecked() },
                }
            }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }

    fn count(self) -> usize {
        self.len()
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining > 0 {
            self.remaining -= 1;

            unsafe {
                match &mut *self.current_back {
                    Entry::Occupied(value) => {
                        let key = self.current_back.offset_from(self.origin) as usize;

                        self.current_back = self.current_back.offset(-1);

                        Some((key, value))
                    },
                    Entry::FreeBlockHead(_) => {
                        let entry = self.current_back.offset(-1);
                        let key = entry.offset_from(self.origin) as usize;

                        self.current_back = entry.offset(-1);

                        if let Entry::Occupied(value) = &mut *entry {
                            Some((key, value))
                        } else {
                            unsafe { unreachable_unchecked() }
                        }
                    },
                    Entry::FreeBlockTail(head_index) => {
                        let end = *head_index - 1;
                        let entry = self.origin.offset(end as isize);

                        self.current_back = entry.offset(-1);

                        if let Entry::Occupied(value) = &mut *entry {
                            Some((end, value))
                        } else {
                            unsafe { unreachable_unchecked() }
                        }
                    }
                }
            }
        } else {
            None
        }
    }
}

impl<'a, T> ExactSizeIterator for IterMut<'a, T> {
    fn len(&self) -> usize {
        self.remaining
    }
}

pub struct Drain<'a, T> {
    slab: &'a mut HopSlab<T>,
    origin: *mut Entry<T>,
    current_front: *mut Entry<T>,
    current_back: *mut Entry<T>,
    remaining: usize,
}

impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining > 0 {
            self.remaining -= 1;

            unsafe {
                let entry = &mut *self.current_front;

                match entry {
                    Entry::Occupied(_) => {
                        self.current_front = self.current_front.offset(1);

                        let entry = mem::replace(entry, Entry::FreeBlockTail(0));

                        if let Entry::Occupied(value) = entry {
                            Some(value)
                        } else {
                            unsafe { unreachable_unchecked() }
                        }
                    }
                    Entry::FreeBlockHead(block) => {
                        let end = block.end;
                        let entry = self.origin.offset(end as isize);

                        self.current_front = entry.offset(1);

                        let entry = mem::replace(&mut *entry, Entry::FreeBlockTail(0));

                        if let Entry::Occupied(value) = entry {
                            Some(value)
                        } else {
                            unsafe { unreachable_unchecked() }
                        }
                    }
                    _ => unsafe { unreachable_unchecked() },
                }
            }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }

    fn count(self) -> usize {
        self.len()
    }
}

impl<'a, T> DoubleEndedIterator for Drain<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining > 0 {
            self.remaining -= 1;

            unsafe {
                let entry = &mut *self.current_back;

                match entry {
                    Entry::Occupied(_) => {
                        self.current_back = self.current_back.offset(-1);

                        let entry = mem::replace(entry, Entry::FreeBlockTail(0));

                        if let Entry::Occupied(value) = entry {
                            Some(value)
                        } else {
                            unsafe { unreachable_unchecked() }
                        }
                    }
                    Entry::FreeBlockHead(_) => {
                        let entry = self.current_back.offset(-1);

                        self.current_back = entry.offset(-1);

                        let entry = mem::replace(&mut *entry, Entry::FreeBlockTail(0));

                        if let Entry::Occupied(value) = entry {
                            Some(value)
                        } else {
                            unsafe { unreachable_unchecked() }
                        }
                    }
                    Entry::FreeBlockTail(head_index) => {
                        let entry = self.origin.offset(*head_index as isize - 1);

                        self.current_back = entry.offset(-1);

                        let entry = mem::replace(&mut *entry, Entry::FreeBlockTail(0));

                        if let Entry::Occupied(value) = entry {
                            Some(value)
                        } else {
                            unsafe { unreachable_unchecked() }
                        }
                    }
                }
            }
        } else {
            None
        }
    }
}

impl<'a, T> ExactSizeIterator for Drain<'a, T> {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        if mem::needs_drop::<T>() {
            // Drop the remaining occupied entries
            while let Some(value) = self.next() {}
        }

        unsafe {
            self.slab.entries.set_len(1);
        }

        self.slab.len = 0;
        self.slab.first_free_block = None;
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[test]
    fn insert_remove() {
        let mut slab = HopSlab::new();

        let a = slab.insert("a");
        let b = slab.insert("b");
        let c = slab.insert("c");
        let d = slab.insert("d");
        let e = slab.insert("e");
        let f = slab.insert("f");
        let g = slab.insert("g");

        assert_eq!(slab.len(), 7);
        assert_eq!(
            slab.entries,
            vec![
                Entry::FreeBlockTail(0), // Sentinel
                Entry::Occupied("a"),
                Entry::Occupied("b"),
                Entry::Occupied("c"),
                Entry::Occupied("d"),
                Entry::Occupied("e"),
                Entry::Occupied("f"),
                Entry::Occupied("g"),
            ]
        );
        assert_eq!(slab.first_free_block, None);

        assert_eq!(slab.remove(a), Some("a"));

        assert_eq!(slab.len(), 6);
        assert_eq!(
            slab.entries,
            vec![
                Entry::FreeBlockTail(0), // Sentinel
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: None,
                    next: None,
                    end: 2
                }),
                Entry::Occupied("b"),
                Entry::Occupied("c"),
                Entry::Occupied("d"),
                Entry::Occupied("e"),
                Entry::Occupied("f"),
                Entry::Occupied("g"),
            ]
        );
        assert_eq!(
            slab.first_free_block,
            Some(unsafe { NonZeroUsize::new_unchecked(1) })
        );

        assert_eq!(slab.remove(a), None);

        assert_eq!(slab.remove(b), Some("b"));

        assert_eq!(slab.len(), 5);
        assert_eq!(
            slab.entries,
            vec![
                Entry::FreeBlockTail(0), // Sentinel
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: None,
                    next: None,
                    end: 3
                }),
                Entry::FreeBlockTail(1),
                Entry::Occupied("c"),
                Entry::Occupied("d"),
                Entry::Occupied("e"),
                Entry::Occupied("f"),
                Entry::Occupied("g"),
            ]
        );
        assert_eq!(
            slab.first_free_block,
            Some(unsafe { NonZeroUsize::new_unchecked(1) })
        );

        assert_eq!(slab.remove(e), Some("e"));

        assert_eq!(slab.len(), 4);
        assert_eq!(
            slab.entries,
            vec![
                Entry::FreeBlockTail(0), // Sentinel
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: Some(unsafe { NonZeroUsize::new_unchecked(5) }),
                    next: None,
                    end: 3
                }),
                Entry::FreeBlockTail(1),
                Entry::Occupied("c"),
                Entry::Occupied("d"),
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: None,
                    next: Some(unsafe { NonZeroUsize::new_unchecked(1) }),
                    end: 6
                }),
                Entry::Occupied("f"),
                Entry::Occupied("g"),
            ]
        );
        assert_eq!(
            slab.first_free_block,
            Some(unsafe { NonZeroUsize::new_unchecked(5) })
        );

        let h = slab.insert("h");

        assert_eq!(slab.len(), 5);
        assert_eq!(
            slab.entries,
            vec![
                Entry::FreeBlockTail(0), // Sentinel
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: None,
                    next: None,
                    end: 3
                }),
                Entry::FreeBlockTail(1),
                Entry::Occupied("c"),
                Entry::Occupied("d"),
                Entry::Occupied("h"),
                Entry::Occupied("f"),
                Entry::Occupied("g"),
            ]
        );
        assert_eq!(
            slab.first_free_block,
            Some(unsafe { NonZeroUsize::new_unchecked(1) })
        );

        assert_eq!(slab.remove(h), Some("h"));

        assert_eq!(slab.len(), 4);
        assert_eq!(
            slab.entries,
            vec![
                Entry::FreeBlockTail(0), // Sentinel
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: Some(unsafe { NonZeroUsize::new_unchecked(5) }),
                    next: None,
                    end: 3
                }),
                Entry::FreeBlockTail(1),
                Entry::Occupied("c"),
                Entry::Occupied("d"),
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: None,
                    next: Some(unsafe { NonZeroUsize::new_unchecked(1) }),
                    end: 6
                }),
                Entry::Occupied("f"),
                Entry::Occupied("g"),
            ]
        );
        assert_eq!(
            slab.first_free_block,
            Some(unsafe { NonZeroUsize::new_unchecked(5) })
        );

        assert_eq!(slab.remove(d), Some("d"));

        assert_eq!(slab.len(), 3);
        assert_eq!(
            slab.entries,
            vec![
                Entry::FreeBlockTail(0), // Sentinel
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: Some(unsafe { NonZeroUsize::new_unchecked(5) }),
                    next: None,
                    end: 3
                }),
                Entry::FreeBlockTail(1),
                Entry::Occupied("c"),
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: None,
                    next: Some(unsafe { NonZeroUsize::new_unchecked(1) }),
                    end: 6
                }),
                Entry::FreeBlockTail(4),
                Entry::Occupied("f"),
                Entry::Occupied("g"),
            ]
        );
        assert_eq!(
            slab.first_free_block,
            Some(unsafe { NonZeroUsize::new_unchecked(5) })
        );

        assert_eq!(slab.remove(f), Some("f"));

        assert_eq!(slab.len(), 2);
        assert_eq!(
            slab.entries,
            vec![
                Entry::FreeBlockTail(0), // Sentinel
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: Some(unsafe { NonZeroUsize::new_unchecked(5) }),
                    next: None,
                    end: 3
                }),
                Entry::FreeBlockTail(1),
                Entry::Occupied("c"),
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: None,
                    next: Some(unsafe { NonZeroUsize::new_unchecked(1) }),
                    end: 7
                }),
                Entry::FreeBlockTail(4),
                Entry::FreeBlockTail(4),
                Entry::Occupied("g"),
            ]
        );
        assert_eq!(
            slab.first_free_block,
            Some(unsafe { NonZeroUsize::new_unchecked(5) })
        );

        assert_eq!(slab.remove(g), Some("g"));

        assert_eq!(slab.len(), 1);
        assert_eq!(
            slab.entries,
            vec![
                Entry::FreeBlockTail(0), // Sentinel
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: Some(unsafe { NonZeroUsize::new_unchecked(5) }),
                    next: None,
                    end: 3
                }),
                Entry::FreeBlockTail(1),
                Entry::Occupied("c"),
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: None,
                    next: Some(unsafe { NonZeroUsize::new_unchecked(1) }),
                    end: 7
                }),
                Entry::FreeBlockTail(4),
                Entry::FreeBlockTail(4),
            ]
        );
        assert_eq!(
            slab.first_free_block,
            Some(unsafe { NonZeroUsize::new_unchecked(5) })
        );

        assert_eq!(slab.remove(c), Some("c"));

        assert_eq!(slab.len(), 0);
        assert_eq!(
            slab.entries,
            vec![
                Entry::FreeBlockTail(0), // Sentinel
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: None,
                    next: None,
                    end: 7
                }),
                Entry::FreeBlockTail(1),
                Entry::FreeBlockTail(1),
                Entry::FreeBlockTail(1),
                Entry::FreeBlockTail(4),
                Entry::FreeBlockTail(1),
            ]
        );
        assert_eq!(
            slab.first_free_block,
            Some(unsafe { NonZeroUsize::new_unchecked(1) })
        );

        slab.insert("a");

        assert_eq!(slab.len(), 1);
        assert_eq!(
            slab.entries,
            vec![
                Entry::FreeBlockTail(0), // Sentinel
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: None,
                    next: None,
                    end: 6
                }),
                Entry::FreeBlockTail(1),
                Entry::FreeBlockTail(1),
                Entry::FreeBlockTail(1),
                Entry::FreeBlockTail(1),
                Entry::Occupied("a"),
            ]
        );
        assert_eq!(
            slab.first_free_block,
            Some(unsafe { NonZeroUsize::new_unchecked(1) })
        );

        slab.insert("b");
        slab.insert("c");
        slab.insert("d");
        slab.insert("e");

        assert_eq!(slab.len(), 5);
        assert_eq!(
            slab.entries,
            vec![
                Entry::FreeBlockTail(0), // Sentinel
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: None,
                    next: None,
                    end: 2
                }),
                Entry::Occupied("e"),
                Entry::Occupied("d"),
                Entry::Occupied("c"),
                Entry::Occupied("b"),
                Entry::Occupied("a"),
            ]
        );
        assert_eq!(
            slab.first_free_block,
            Some(unsafe { NonZeroUsize::new_unchecked(1) })
        );

        slab.insert("f");

        assert_eq!(slab.len(), 6);
        assert_eq!(
            slab.entries,
            vec![
                Entry::FreeBlockTail(0), // Sentinel
                Entry::Occupied("f"),
                Entry::Occupied("e"),
                Entry::Occupied("d"),
                Entry::Occupied("c"),
                Entry::Occupied("b"),
                Entry::Occupied("a"),
            ]
        );
        assert_eq!(slab.first_free_block, None);

        slab.insert("g");

        assert_eq!(slab.len(), 7);
        assert_eq!(
            slab.entries,
            vec![
                Entry::FreeBlockTail(0), // Sentinel
                Entry::Occupied("f"),
                Entry::Occupied("e"),
                Entry::Occupied("d"),
                Entry::Occupied("c"),
                Entry::Occupied("b"),
                Entry::Occupied("a"),
                Entry::Occupied("g"),
            ]
        );
        assert_eq!(slab.first_free_block, None);
    }

    #[test]
    fn compact() {
        let mut slab = HopSlab::new();

        let a = slab.insert("a");
        let _b = slab.insert("b");
        let c = slab.insert("c");
        let d = slab.insert("d");
        let _e = slab.insert("e");
        let f = slab.insert("f");
        let _g = slab.insert("g");
        let h = slab.insert("h");
        let _i = slab.insert("i");
        let j = slab.insert("j");

        slab.remove(a);
        slab.remove(c);
        slab.remove(d);
        slab.remove(f);
        slab.remove(h);
        slab.remove(j);

        let prior_capacity = slab.entries.capacity();

        slab.compact(|_, _, _| true);

        assert_eq!(
            slab.entries,
            vec![
                Entry::FreeBlockTail(0),
                Entry::Occupied("b"),
                Entry::Occupied("e"),
                Entry::Occupied("g"),
                Entry::Occupied("i"),
            ]
        );
        assert_eq!(slab.first_free_block, None);
        assert_eq!(slab.entries.capacity(), prior_capacity);
    }

    #[test]
    fn compact_abort() {
        let mut slab = HopSlab::new();

        let a = slab.insert("a");
        let _b = slab.insert("b");
        let c = slab.insert("c");
        let d = slab.insert("d");
        let _e = slab.insert("e");
        let f = slab.insert("f");
        let g = slab.insert("g");
        let h = slab.insert("h");
        let _i = slab.insert("i");
        let j = slab.insert("j");

        slab.remove(a);
        slab.remove(c);
        slab.remove(d);
        slab.remove(f);
        slab.remove(h);
        slab.remove(j);

        let prior_capacity = slab.entries.capacity();

        slab.compact(|_, old_key, _| old_key != g);

        assert_eq!(
            slab.entries,
            vec![
                Entry::FreeBlockTail(0),
                Entry::Occupied("b"),
                Entry::Occupied("e"),
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: Some(unsafe { NonZeroUsize::new_unchecked(8) }),
                    next: None,
                    end: 7
                }),
                Entry::FreeBlockTail(3),
                Entry::FreeBlockTail(0),
                Entry::FreeBlockTail(3),
                Entry::Occupied("g"),
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: None,
                    next: Some(unsafe { NonZeroUsize::new_unchecked(3) }),
                    end: 9
                }),
                Entry::Occupied("i"),
            ]
        );
        assert_eq!(
            slab.first_free_block,
            Some(unsafe { NonZeroUsize::new_unchecked(8) })
        );
        assert_eq!(slab.entries.capacity(), prior_capacity);
    }

    #[test]
    fn compact_panic() {
        let mut slab = HopSlab::new();

        let a = slab.insert("a");
        let _b = slab.insert("b");
        let c = slab.insert("c");
        let d = slab.insert("d");
        let _e = slab.insert("e");
        let f = slab.insert("f");
        let g = slab.insert("g");
        let h = slab.insert("h");
        let _i = slab.insert("i");
        let j = slab.insert("j");

        slab.remove(a);
        slab.remove(c);
        slab.remove(d);
        slab.remove(f);
        slab.remove(h);
        slab.remove(j);

        let prior_capacity = slab.entries.capacity();

        catch_unwind(AssertUnwindSafe(|| {
            slab.compact(|_, old_key, _| {
                if old_key == g {
                    panic!();
                }

                true
            });
        }));

        assert_eq!(
            slab.entries,
            vec![
                Entry::FreeBlockTail(0),
                Entry::Occupied("b"),
                Entry::Occupied("e"),
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: Some(unsafe { NonZeroUsize::new_unchecked(8) }),
                    next: None,
                    end: 7
                }),
                Entry::FreeBlockTail(3),
                Entry::FreeBlockTail(0),
                Entry::FreeBlockTail(3),
                Entry::Occupied("g"),
                Entry::FreeBlockHead(FreeBlockHead {
                    previous: None,
                    next: Some(unsafe { NonZeroUsize::new_unchecked(3) }),
                    end: 9
                }),
                Entry::Occupied("i"),
            ]
        );
        assert_eq!(
            slab.first_free_block,
            Some(unsafe { NonZeroUsize::new_unchecked(8) })
        );
        assert_eq!(slab.entries.capacity(), prior_capacity);
    }
}
