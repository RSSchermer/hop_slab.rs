use hop_slab::*;

use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

#[test]
fn insert_get_remove_get() {
    let mut slab = HopSlab::new();

    let a = slab.insert("a");

    assert_eq!(slab.get(a), Some(&"a"));
    assert_eq!(slab.remove(a), Some("a"));
    assert_eq!(slab.get(a), None);
}

#[test]
fn compact_shrink_to_fit() {
    let mut slab = HopSlab::new();

    let a = slab.insert("a");
    let b = slab.insert("b");
    let c = slab.insert("c");
    let d = slab.insert("d");
    let e = slab.insert("e");
    let f = slab.insert("f");
    let g = slab.insert("g");

    slab.remove(b);
    slab.remove(c);
    slab.remove(f);

    slab.compact(|_, _, _| true);
    slab.shrink_to_fit();

    assert_eq!(slab.capacity(), 4);
    assert_eq!(
        slab.into_iter().collect::<Vec<_>>(),
        vec![(1, "a"), (2, "d"), (3, "e"), (4, "g")]
    );
}

#[test]
fn compact_abort() {
    let mut slab = HopSlab::new();

    let a = slab.insert("a");
    let b = slab.insert("b");
    let c = slab.insert("c");
    let d = slab.insert("d");
    let e = slab.insert("e");
    let f = slab.insert("f");
    let g = slab.insert("g");

    slab.remove(b);
    slab.remove(c);
    slab.remove(f);

    slab.compact(|_, old_key, _| old_key != e);

    assert_eq!(
        slab.into_iter().collect::<Vec<_>>(),
        vec![(1, "a"), (2, "d"), (5, "e"), (7, "g")]
    );
}

#[test]
fn compact_panic() {
    let mut slab = HopSlab::new();

    let a = slab.insert("a");
    let b = slab.insert("b");
    let c = slab.insert("c");
    let d = slab.insert("d");
    let e = slab.insert("e");
    let f = slab.insert("f");
    let g = slab.insert("g");

    slab.remove(b);
    slab.remove(c);
    slab.remove(f);

    catch_unwind(AssertUnwindSafe(|| {
        slab.compact(|_, old_key, _| {
            if old_key == e {
                panic!();
            }

            true
        });
    }));

    assert_eq!(
        slab.into_iter().collect::<Vec<_>>(),
        vec![(1, "a"), (2, "d"), (5, "e"), (7, "g")]
    );
}

#[test]
fn vacant_entry() {
    let mut slab = HopSlab::new();

    let a = slab.insert("a");
    let b = slab.insert("b");
    let c = slab.insert("c");

    slab.remove(b);

    let vacant_entry = slab.vacant_entry();

    assert_eq!(vacant_entry.key(), b);

    vacant_entry.insert("d");

    assert_eq!(slab.get(b), Some(&"d"));
}

#[test]
fn retain() {
    let mut slab = HopSlab::new();

    let a = slab.insert("a");
    let b = slab.insert("b");
    let c = slab.insert("c");
    let d = slab.insert("d");
    let e = slab.insert("e");
    let f = slab.insert("f");
    let g = slab.insert("g");

    // Retain even keys. Note that keys start at 1 (because of a sentinel entry).
    slab.retain(|key, _value| key % 2 == 0);

    assert_eq!(slab.get(a), None);
    assert_eq!(slab.get(b), Some(&"b"));
    assert_eq!(slab.get(c), None);
    assert_eq!(slab.get(d), Some(&"d"));
    assert_eq!(slab.get(e), None);
    assert_eq!(slab.get(f), Some(&"f"));
    assert_eq!(slab.get(g), None);
}

#[test]
fn drain() {
    let mut slab = HopSlab::new();

    let a = slab.insert("a");
    let b = slab.insert("b");
    let c = slab.insert("c");
    let d = slab.insert("d");
    let e = slab.insert("e");
    let f = slab.insert("f");
    let g = slab.insert("g");

    slab.remove(b);
    slab.remove(c);
    slab.remove(f);

    let prior_capacity = slab.capacity();
    let drain = slab.drain();

    assert_eq!(drain.collect::<Vec<_>>(), vec!["a", "d", "e", "g"]);
    assert!(slab.is_empty());
    assert_eq!(slab.capacity(), prior_capacity);
}

#[test]
fn drain_rev() {
    let mut slab = HopSlab::new();

    let a = slab.insert("a");
    let b = slab.insert("b");
    let c = slab.insert("c");
    let d = slab.insert("d");
    let e = slab.insert("e");
    let f = slab.insert("f");
    let g = slab.insert("g");

    slab.remove(b);
    slab.remove(c);
    slab.remove(f);

    let prior_capacity = slab.capacity();
    let drain = slab.drain();

    assert_eq!(drain.rev().collect::<Vec<_>>(), vec!["g", "e", "d", "a"]);
    assert!(slab.is_empty());
    assert_eq!(slab.capacity(), prior_capacity);
}

#[test]
fn into_iter() {
    let mut slab = HopSlab::new();

    let a = slab.insert("a");
    let b = slab.insert("b");
    let c = slab.insert("c");
    let d = slab.insert("d");
    let e = slab.insert("e");
    let f = slab.insert("f");
    let g = slab.insert("g");

    slab.remove(b);
    slab.remove(c);
    slab.remove(f);

    let mut iter = slab.into_iter();

    assert_eq!(iter.next(), Some((a, "a")));
    assert_eq!(iter.next(), Some((d, "d")));
    assert_eq!(iter.next(), Some((e, "e")));
    assert_eq!(iter.next(), Some((g, "g")));
    assert_eq!(iter.next(), None);
}

#[test]
fn into_iter_rev() {
    let mut slab = HopSlab::new();

    let a = slab.insert("a");
    let b = slab.insert("b");
    let c = slab.insert("c");
    let d = slab.insert("d");
    let e = slab.insert("e");
    let f = slab.insert("f");
    let g = slab.insert("g");

    slab.remove(b);
    slab.remove(c);
    slab.remove(f);

    let mut iter = slab.into_iter().rev();

    assert_eq!(iter.next(), Some((g, "g")));
    assert_eq!(iter.next(), Some((e, "e")));
    assert_eq!(iter.next(), Some((d, "d")));
    assert_eq!(iter.next(), Some((a, "a")));
    assert_eq!(iter.next(), None);
}

#[test]
fn into_iter_alternate_front_back() {
    let mut slab = HopSlab::new();

    let a = slab.insert("a");
    let b = slab.insert("b");
    let c = slab.insert("c");
    let d = slab.insert("d");
    let e = slab.insert("e");
    let f = slab.insert("f");
    let g = slab.insert("g");

    slab.remove(b);
    slab.remove(c);
    slab.remove(f);

    let mut iter = slab.into_iter();

    assert_eq!(iter.next(), Some((a, "a")));
    assert_eq!(iter.next_back(), Some((g, "g")));
    assert_eq!(iter.next(), Some((d, "d")));
    assert_eq!(iter.next_back(), Some((e, "e")));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next_back(), None);
}

#[test]
fn iter() {
    let mut slab = HopSlab::new();

    let a = slab.insert("a");
    let b = slab.insert("b");
    let c = slab.insert("c");
    let d = slab.insert("d");
    let e = slab.insert("e");
    let f = slab.insert("f");
    let g = slab.insert("g");

    slab.remove(b);
    slab.remove(c);
    slab.remove(f);

    let mut iter = slab.iter();

    assert_eq!(iter.next(), Some((a, &"a")));
    assert_eq!(iter.next(), Some((d, &"d")));
    assert_eq!(iter.next(), Some((e, &"e")));
    assert_eq!(iter.next(), Some((g, &"g")));
    assert_eq!(iter.next(), None);
}

#[test]
fn iter_rev() {
    let mut slab = HopSlab::new();

    let a = slab.insert("a");
    let b = slab.insert("b");
    let c = slab.insert("c");
    let d = slab.insert("d");
    let e = slab.insert("e");
    let f = slab.insert("f");
    let g = slab.insert("g");

    slab.remove(b);
    slab.remove(c);
    slab.remove(f);

    let mut iter = slab.iter().rev();

    assert_eq!(iter.next(), Some((g, &"g")));
    assert_eq!(iter.next(), Some((e, &"e")));
    assert_eq!(iter.next(), Some((d, &"d")));
    assert_eq!(iter.next(), Some((a, &"a")));
    assert_eq!(iter.next(), None);
}

#[test]
fn iter_alternate_front_back() {
    let mut slab = HopSlab::new();

    let a = slab.insert("a");
    let b = slab.insert("b");
    let c = slab.insert("c");
    let d = slab.insert("d");
    let e = slab.insert("e");
    let f = slab.insert("f");
    let g = slab.insert("g");

    slab.remove(b);
    slab.remove(c);
    slab.remove(f);

    let mut iter = slab.iter();

    assert_eq!(iter.next(), Some((a, &"a")));
    assert_eq!(iter.next_back(), Some((g, &"g")));
    assert_eq!(iter.next(), Some((d, &"d")));
    assert_eq!(iter.next_back(), Some((e, &"e")));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next_back(), None);
}

#[test]
fn iter_mut() {
    let mut slab = HopSlab::new();

    let a = slab.insert("a");
    let b = slab.insert("b");
    let c = slab.insert("c");
    let d = slab.insert("d");
    let e = slab.insert("e");
    let f = slab.insert("f");
    let g = slab.insert("g");

    slab.remove(b);
    slab.remove(c);
    slab.remove(f);

    let mut iter = slab.iter_mut();

    assert_eq!(iter.next(), Some((a, &mut "a")));
    assert_eq!(iter.next(), Some((d, &mut "d")));
    assert_eq!(iter.next(), Some((e, &mut "e")));
    assert_eq!(iter.next(), Some((g, &mut "g")));
    assert_eq!(iter.next(), None);
}

#[test]
fn iter_mut_rev() {
    let mut slab = HopSlab::new();

    let a = slab.insert("a");
    let b = slab.insert("b");
    let c = slab.insert("c");
    let d = slab.insert("d");
    let e = slab.insert("e");
    let f = slab.insert("f");
    let g = slab.insert("g");

    slab.remove(b);
    slab.remove(c);
    slab.remove(f);

    let mut iter = slab.iter_mut().rev();

    assert_eq!(iter.next(), Some((g, &mut "g")));
    assert_eq!(iter.next(), Some((e, &mut "e")));
    assert_eq!(iter.next(), Some((d, &mut "d")));
    assert_eq!(iter.next(), Some((a, &mut "a")));
    assert_eq!(iter.next(), None);
}

#[test]
fn iter_mut_alternate_front_back() {
    let mut slab = HopSlab::new();

    let a = slab.insert("a");
    let b = slab.insert("b");
    let c = slab.insert("c");
    let d = slab.insert("d");
    let e = slab.insert("e");
    let f = slab.insert("f");
    let g = slab.insert("g");

    slab.remove(b);
    slab.remove(c);
    slab.remove(f);

    let mut iter = slab.iter_mut();

    assert_eq!(iter.next(), Some((a, &mut "a")));
    assert_eq!(iter.next_back(), Some((g, &mut "g")));
    assert_eq!(iter.next(), Some((d, &mut "d")));
    assert_eq!(iter.next_back(), Some((e, &mut "e")));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next_back(), None);
}
