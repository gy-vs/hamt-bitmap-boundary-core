import { describe, expect, it } from 'vitest';
import { PersistentMap, hashKey } from '../src/index.js';

const UINT32_MAX = 4294967295; // 0xFFFFFFFF

const keysOf = <V>(map: PersistentMap<V>) => map.items().map(item => item.key);

const hashesOf = <V>(map: PersistentMap<V>) => map.items().map(item => item.hash);

describe('existing behavior', () => {
  it('persists', () => {
    const a = new PersistentMap<number>();
    const b = a.set('x', 1);
    expect(a.size()).toBe(0);
    expect(b.get('x')).toBe(1);
  });

  it('replaces an existing key without growing', () => {
    const a = new PersistentMap<number>().set('x', 1, 7);
    const b = a.set('x', 2, 7);
    expect(b.size()).toBe(1);
    expect(b.get('x', 7)).toBe(2);
    expect(a.get('x', 7)).toBe(1);
  });

  it('builds from an entries array via the constructor', () => {
    const map = new PersistentMap<number>([
      { key: 'a', value: 1, hash: 31 },
      { key: 'b', value: 2, hash: -1 },
    ]);
    expect(map.size()).toBe(2);
    expect(map.get('a', 31)).toBe(1);
    expect(map.get('b', UINT32_MAX)).toBe(2);
  });

  it('returns undefined for missing keys and misses near existing slots', () => {
    const map = new PersistentMap<number>().set('a', 1, 0).set('b', 2, 31);
    expect(map.get('nope')).toBeUndefined();
    expect(map.get('a', 1)).toBeUndefined(); // same key path, wrong hash
    expect(map.get('c', 0)).toBeUndefined(); // occupied slot, absent key
  });
});

describe('unsigned bitmap slot boundaries', () => {
  it('orders slots 0, 30 and 31 ascending at the root', () => {
    const map = new PersistentMap<number>()
      .set('slot-31', 31, 31)
      .set('slot-0', 0, 0)
      .set('slot-30', 30, 30);
    expect(map.size()).toBe(3);
    // slot 31 must sort after 0 and 30, not before them as a negative sign bit
    expect(keysOf(map)).toEqual(['slot-0', 'slot-30', 'slot-31']);
    expect(map.get('slot-31', 31)).toBe(31);
    expect(map.get('slot-30', 30)).toBe(30);
    expect(map.get('slot-0', 0)).toBe(0);
  });

  it('orders slots 0, 30 and 31 ascending at deeper levels', () => {
    // all share root slot 1; second-level fragments are 0, 30, 31
    const map = new PersistentMap<number>()
      .set('deep-31', 31, (31 << 5) | 1)
      .set('deep-0', 0, 1)
      .set('deep-30', 30, (30 << 5) | 1);
    expect(keysOf(map)).toEqual(['deep-0', 'deep-30', 'deep-31']);
    expect(map.get('deep-31', (31 << 5) | 1)).toBe(31);
    expect(map.get('deep-30', (30 << 5) | 1)).toBe(30);
    expect(map.get('deep-0', 1)).toBe(0);
  });

  it('handles slot 31 at every level via an all-ones hash', () => {
    // 0xFFFFFFFF picks slot 31 on all seven levels
    const map = new PersistentMap<number>()
      .set('ones', 1, UINT32_MAX)
      .set('low', 2, 0x0fffffff); // diverges only at the deepest fragment
    expect(map.get('ones', UINT32_MAX)).toBe(1);
    expect(map.get('low', 0x0fffffff)).toBe(2);
    expect(map.size()).toBe(2);
  });

  it('keeps every key after a full 32-slot bitmap expands to an array node', () => {
    let map = new PersistentMap<number>();
    for (let slot = 0; slot < 32; slot++) map = map.set(`k${slot}`, slot, slot);
    expect(map.size()).toBe(32);
    for (let slot = 0; slot < 32; slot++) expect(map.get(`k${slot}`, slot)).toBe(slot);
    expect(hashesOf(map)).toEqual(Array.from({ length: 32 }, (_, slot) => slot));
  });

  it('handles alternating even slots (0x55555555)', () => {
    let map = new PersistentMap<number>();
    for (let slot = 0; slot < 32; slot += 2) map = map.set(`even-${slot}`, slot, slot);
    expect(map.size()).toBe(16);
    for (let slot = 0; slot < 32; slot += 2) {
      expect(map.get(`even-${slot}`, slot)).toBe(slot);
    }
    expect(keysOf(map)).toEqual(Array.from({ length: 16 }, (_, i) => `even-${i * 2}`));
  });

  it('handles alternating odd slots (0xAAAAAAAA, sign bit set)', () => {
    let map = new PersistentMap<number>();
    for (let slot = 1; slot < 32; slot += 2) map = map.set(`odd-${slot}`, slot, slot);
    expect(map.size()).toBe(16);
    for (let slot = 1; slot < 32; slot += 2) {
      expect(map.get(`odd-${slot}`, slot)).toBe(slot);
    }
    // slot 31 participates and stays last
    expect(keysOf(map)[15]).toBe('odd-31');
    expect(map.get('odd-31', 31)).toBe(31);
  });

  it('keeps alternating keys when the 17th slot forces expansion', () => {
    let map = new PersistentMap<number>();
    for (let slot = 0; slot < 32; slot += 2) map = map.set(`even-${slot}`, slot, slot);
    map = map.set('extra', 100, 1); // 17th distinct slot -> array node
    expect(map.size()).toBe(17);
    for (let slot = 0; slot < 32; slot += 2) {
      expect(map.get(`even-${slot}`, slot)).toBe(slot);
    }
    expect(map.get('extra', 1)).toBe(100);
    expect(keysOf(map)[1]).toBe('extra'); // slot 1 sorts between slots 0 and 2
  });
});

describe('hash normalization', () => {
  it('treats negative JavaScript number hashes as uint32', () => {
    const map = new PersistentMap<number>().set('neg', 42, -1);
    expect(map.get('neg', -1)).toBe(42);
    expect(map.get('neg', UINT32_MAX)).toBe(42); // -1 >>> 0 === 0xFFFFFFFF
    expect(map.items()[0].hash).toBe(UINT32_MAX); // stored hash stays unsigned
    expect(map.delete('neg', -1).size()).toBe(0);
  });

  it('normalizes other negative hashes consistently between set/get/delete', () => {
    const hash = -2147483648; // INT32_MIN -> 0x80000000, root slot 0, then slot 0... deep
    const map = new PersistentMap<number>().set('min', 7, hash);
    expect(map.get('min', hash)).toBe(7);
    expect(map.get('min', 2147483648)).toBe(7);
    expect(map.items()[0].hash).toBe(2147483648);
    expect(map.delete('min', 2147483648).get('min', hash)).toBeUndefined();
  });

  it('supports the maximum uint32 custom hash with collisions sorted canonically', () => {
    const map = new PersistentMap<number>()
      .set('c', 3, UINT32_MAX)
      .set('a', 1, UINT32_MAX)
      .set('b', 2, UINT32_MAX);
    expect(map.size()).toBe(3);
    expect(map.get('a', UINT32_MAX)).toBe(1);
    expect(map.get('b', UINT32_MAX)).toBe(2);
    expect(map.get('c', UINT32_MAX)).toBe(3);
    // same full hash => collision node, traversed in key order regardless of insertion order
    expect(keysOf(map)).toEqual(['a', 'b', 'c']);

    const afterDelete = map.delete('b', UINT32_MAX);
    expect(afterDelete.size()).toBe(2);
    expect(keysOf(afterDelete)).toEqual(['a', 'c']);
    expect(afterDelete.get('b', UINT32_MAX)).toBeUndefined();

    const one = afterDelete.delete('a', UINT32_MAX);
    expect(one.size()).toBe(1);
    expect(one.get('c', UINT32_MAX)).toBe(3);
    expect(one.delete('c', UINT32_MAX).size()).toBe(0);
  });
});

describe('expansion and contraction across the threshold', () => {
  it('survives repeated grow/shrink cycles without losing keys', () => {
    let map = new PersistentMap<number>();
    for (let round = 0; round < 3; round++) {
      for (let slot = 0; slot < 20; slot++) map = map.set(`k${slot}`, round * 100 + slot, slot);
      expect(map.size()).toBe(20);
      for (let slot = 0; slot < 20; slot++) {
        expect(map.get(`k${slot}`, slot)).toBe(round * 100 + slot);
      }
      for (let slot = 0; slot < 15; slot++) map = map.delete(`k${slot}`, slot);
      expect(map.size()).toBe(5);
      for (let slot = 15; slot < 20; slot++) {
        expect(map.get(`k${slot}`, slot)).toBe(round * 100 + slot);
      }
      expect(keysOf(map)).toEqual(['k15', 'k16', 'k17', 'k18', 'k19']);
      for (let slot = 15; slot < 20; slot++) map = map.delete(`k${slot}`, slot);
      expect(map.size()).toBe(0);
    }
  });

  it('uses the same slot order on both sides of the threshold', () => {
    let map = new PersistentMap<number>();
    for (let slot = 0; slot < 17; slot++) map = map.set(`k${slot}`, slot, slot); // array node
    const expandedOrder = keysOf(map);
    map = map.delete('k16', 16); // 16 left, still array
    map = map.delete('k15', 15);
    map = map.delete('k14', 14);
    map = map.delete('k13', 13);
    map = map.delete('k12', 12);
    map = map.delete('k11', 11);
    map = map.delete('k10', 10);
    map = map.delete('k9', 9);
    map = map.delete('k8', 8); // 8 left -> shrink back to bitmap node
    expect(map.size()).toBe(8);
    expect(keysOf(map)).toEqual(expandedOrder.slice(0, 8));
    for (let slot = 0; slot < 8; slot++) expect(map.get(`k${slot}`, slot)).toBe(slot);
  });
});

describe('canonical results independent of insertion order', () => {
  // root slots 0..17 plus 30/31 (forces array node), deep splits under slot 1,
  // and a same-hash collision trio at 0xFFFFFFFF
  const pairs: [string, number][] = [
    ['deep-lo', 1],
    ['deep-hi-30', (30 << 5) | 1],
    ['deep-hi-31', (31 << 5) | 1],
    ['tail-30', 30],
    ['tail-31', 31],
    ['max-a', UINT32_MAX],
    ['max-b', UINT32_MAX],
    ['max-c', UINT32_MAX],
    ...Array.from({ length: 16 }, (_, slot): [string, number] => [`slot-${slot}`, slot + 2]),
  ];

  const build = (order: [string, number][]) => {
    let map = new PersistentMap<number>();
    for (const [key, hash] of order) map = map.set(key, hashKey(key), hash);
    return map;
  };

  it('produces identical queries and traversal for different insertion orders', () => {
    const forward = build(pairs);
    const reversed = build([...pairs].reverse());
    const interleaved = build(pairs.filter((_, i) => i % 2 === 0).concat(pairs.filter((_, i) => i % 2 === 1)));

    expect(forward.size()).toBe(pairs.length);
    expect(forward.items()).toEqual(reversed.items());
    expect(forward.items()).toEqual(interleaved.items());
    for (const [key, hash] of pairs) {
      expect(reversed.get(key, hash)).toBe(forward.get(key, hash));
      expect(interleaved.get(key, hash)).toBe(forward.get(key, hash));
    }
  });

  it('matches the canonical order after deletes and reinserts', () => {
    const reference = build(pairs);
    let map = build(pairs);
    for (const [key, hash] of pairs.slice(0, 10)) map = map.delete(key, hash);
    for (const [key, hash] of pairs.slice(0, 10)) map = map.set(key, hashKey(key), hash);
    expect(map.items()).toEqual(reference.items());
  });
});

describe('default string hashing at scale', () => {
  it('stores, reads and deletes many keys through hashKey', () => {
    let map = new PersistentMap<number>();
    const keys = Array.from({ length: 200 }, (_, i) => `key-${i}`);
    for (const [i, key] of keys.entries()) map = map.set(key, i);
    expect(map.size()).toBe(200);
    for (const [i, key] of keys.entries()) expect(map.get(key)).toBe(i);

    const snapshot = map;
    for (const key of keys.slice(0, 100)) map = map.delete(key);
    expect(map.size()).toBe(100);
    expect(snapshot.size()).toBe(200); // persistence: earlier versions are untouched
    for (const key of keys.slice(0, 100)) expect(map.get(key)).toBeUndefined();
    for (const [i, key] of keys.slice(100).entries()) expect(map.get(key)).toBe(i + 100);
  });
});
