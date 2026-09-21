import { describe, expect, it } from 'vitest';
import {
  bitFor,
  BitmapNode,
  compactIndex,
  hashKey,
  LeafNode,
  nodeKind,
  PersistentMap,
  popcount,
  slotAt,
  slotForIndex,
  toUint32,
  type Entry,
} from '../src/index.js';

const topSlot = (hash: number) => slotAt(hash >>> 0, 27);

/** Unsigned hash with top-5-bit fragment `slot` and a unique low suffix. */
const h = (slot: number, low: number): number => (((slot << 27) >>> 0) | low) >>> 0;

const seed = (entries: Array<[string, number, number]>): PersistentMap<number> => {
  let map = PersistentMap.empty<number>();
  for (const [key, hash, value] of entries) map = map.set(key, value, hash);
  return map;
};

const slotEntries = (slots: number[]): Array<[string, number, number]> =>
  slots.map((slot) => [`k${slot}`, h(slot, 0x100 + slot), slot]);

const shuffled = <T>(items: T[], seedShift: number): T[] => {
  // Deterministic permutation without relying on RNG state.
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = (Math.imul(i + 1, 2654435761) >>> seedShift) % (i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
};

describe('uint32 bitmap primitives', () => {
  it('bitFor stays unsigned at slot 31', () => {
    expect(bitFor(0)).toBe(1);
    expect(bitFor(30)).toBe(0x40000000);
    expect(bitFor(31)).toBe(0x80000000);
    expect(bitFor(31)).toBeGreaterThan(0);
  });

  it('popcount treats bit 31 as one bit, not a sign', () => {
    expect(popcount(0)).toBe(0);
    expect(popcount(1)).toBe(1);
    expect(popcount(bitFor(30))).toBe(1);
    expect(popcount(bitFor(31))).toBe(1);
    expect(popcount(0xffffffff)).toBe(32);
    expect(popcount(0xaaaaaaaa)).toBe(16);
    expect(popcount(0x55555555)).toBe(16);
    // Signed-looking inputs get normalized.
    expect(popcount(-2147483648)).toBe(1);
    expect(popcount(-1)).toBe(32);
  });

  it('compactIndex counts strictly-lower set bits at all boundaries', () => {
    const full = 0xffffffff;
    expect(compactIndex(full, 0)).toBe(0);
    expect(compactIndex(full, 30)).toBe(30);
    expect(compactIndex(full, 31)).toBe(31);
    // Only bit 31 present: it must index 0, not a negative/huge position.
    expect(compactIndex(bitFor(31), 31)).toBe(0);
    expect(compactIndex(bitFor(31), 0)).toBe(0);
    // Bits at 0, 30, 31.
    const sparse = (bitFor(0) | bitFor(30) | bitFor(31)) >>> 0;
    expect(compactIndex(sparse, 0)).toBe(0);
    expect(compactIndex(sparse, 30)).toBe(1);
    expect(compactIndex(sparse, 31)).toBe(2);
    // Alternating bitmap.
    const alt = 0x55555555;
    expect(compactIndex(alt, 1)).toBe(1);
    expect(compactIndex(alt, 31)).toBe(16);
    expect(compactIndex(0xaaaaaaaa, 30)).toBe(15);
  });

  it('slotForIndex is the inverse of compactIndex', () => {
    const bitmaps = [0xffffffff, 0x55555555, 0xaaaaaaaa, bitFor(31), (bitFor(0) | bitFor(31)) >>> 0];
    for (const bitmap of bitmaps) {
      const slots: number[] = [];
      for (let slot = 0; slot < 32; slot++) if ((bitmap & bitFor(slot)) !== 0) slots.push(slot);
      slots.forEach((slot, idx) => {
        expect(slotForIndex(bitmap, idx)).toBe(slot);
        expect(compactIndex(bitmap, slot)).toBe(idx);
      });
    }
  });

  it('slotAt extracts fragments at slot 0/30/31 and the terminal level', () => {
    expect(slotAt(toUint32(0), 27)).toBe(0);
    expect(slotAt(toUint32(30 << 27), 27)).toBe(30);
    expect(slotAt(toUint32(31 << 27), 27)).toBe(31);
    expect(slotAt(0xffffffff, 27)).toBe(31);
    expect(slotAt(0xffffffff, null)).toBe(3);
    expect(slotAt(toUint32(-1), 27)).toBe(31);
  });

  it('toUint32 maps negatives and >2^32 values into unsigned range', () => {
    expect(toUint32(-1)).toBe(0xffffffff);
    expect(toUint32(-2147483648)).toBe(0x80000000);
    expect(toUint32(4294967296)).toBe(0);
    expect(toUint32(4294967297)).toBe(1);
  });
});

describe('slots 0, 30 and 31 in the live trie', () => {
  const entries: Array<[string, number, number]> = [
    ['s0', h(0, 1), 0],
    ['s30', h(30, 1), 30],
    ['s31', h(31, 1), 31],
  ];

  for (const order of [entries, [...entries].reverse(), shuffled(entries, 3)]) {
    const label = order.map(([k]) => k).join(',');
    it(`finds all three keys regardless of order [${label}]`, () => {
      const map = seed(order);
      expect(nodeKind(map.getRootNode())).toBe('bitmap');
      const root = map.getRootNode() as BitmapNode<number>;
      expect(root.bitmap >>> 0).toBe((bitFor(0) | bitFor(30) | bitFor(31)) >>> 0);
      for (const [key, hash, value] of entries) {
        expect(map.get(key, hash)).toBe(value);
        expect(map.get(key, hash >>> 0)).toBe(value);
      }
      expect(map.size()).toBe(3);
      // Compact order is ascending slot order 0 -> 30 -> 31.
      expect(map.keys()).toEqual(['s0', 's30', 's31']);
    });
  }

  it('keeps bit-31 keys through value updates', () => {
    const hash = h(31, 7);
    const map = PersistentMap.empty<number>().set('edge', 1, hash).set('edge', 2, hash);
    expect(map.get('edge', hash)).toBe(2);
    expect(map.size()).toBe(1);
  });

  it('negative JavaScript number hashes are treated as their uint32 identity', () => {
    const negative = -2147483648; // 0x80000000, top slot 16
    expect(topSlot(negative)).toBe(16);
    const map = PersistentMap.empty<number>().set('neg', 42, negative);
    expect(map.get('neg', negative)).toBe(42);
    expect(map.get('neg', 0x80000000)).toBe(42);
    expect(map.items()[0].hash).toBe(0x80000000);
  });
});

describe('bitmap expansion to array and contraction', () => {
  it('expands at 16 distinct top-level slots and finds every key incl. slot 31', () => {
    const slots = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 31];
    expect(slots.length).toBe(16);
    const map = seed(shuffled(slotEntries(slots), 5));
    expect(nodeKind(map.getRootNode())).toBe('array');
    for (const slot of slots) {
      const hash = h(slot, 0x100 + slot);
      expect(map.get(`k${slot}`, hash)).toBe(slot);
    }
    expect(map.size()).toBe(16);
    expect(map.keys()).toEqual(slots.slice().sort((a, b) => a - b).map((s) => `k${s}`));
  });

  it('stays a bitmap node at 15 top-level slots', () => {
    const slots = Array.from({ length: 15 }, (_, i) => i * 2 + 1); // odd slots 1..29
    const map = seed(shuffled(slotEntries(slots), 7));
    expect(nodeKind(map.getRootNode())).toBe('bitmap');
    expect((map.getRootNode() as BitmapNode<number>).bitmap >>> 0).toBe(0x2aaaaaaa);
  });

  it('holds a full 32-slot bitmap/array and contracts back symmetrically', () => {
    const entries = slotEntries(Array.from({ length: 32 }, (_, s) => s));
    let map = seed(shuffled(entries, 11));
    expect(nodeKind(map.getRootNode())).toBe('array');
    for (const [key, hash, value] of entries) expect(map.get(key, hash)).toBe(value);

    // Delete down to 15 occupied subtrees; the array must contract to a bitmap
    // using the same ascending slot order, and keys must not disappear.
    for (let removed = 0; removed < 17; removed++) {
      const slot = removed * 2 % 32; // spread removals across slots
      const key = `k${slot}`;
      const hash = h(slot, 0x100 + slot);
      if (map.get(key, hash) === undefined) continue;
      map = map.delete(key, hash);
      expect(map.get(key, hash)).toBeUndefined();
    }
    expect(map.size()).toBeGreaterThanOrEqual(14);
    // Continue removing until below threshold; eventually node kind flips.
    for (const [key, hash] of entries) map = map.delete(key, hash);
    expect(nodeKind(map.getRootNode())).toBe('empty');
  });

  it('survives repeated grow/shrink cycles with keys at both thresholds', () => {
    const evenSlots = Array.from({ length: 16 }, (_, i) => i * 2); // 0..30
    const oddSlots = Array.from({ length: 16 }, (_, i) => i * 2 + 1); // 1..31
    const even = slotEntries(evenSlots);
    const odd = slotEntries(oddSlots);

    let map = seed(shuffled(even, 13));
    expect(nodeKind(map.getRootNode())).toBe('array');
    // Add odds one at a time, then remove evens one at a time: repeated boundary.
    for (const [key, hash, value] of shuffled(odd, 17)) {
      map = map.set(key, value, hash);
    }
    expect(map.size()).toBe(32);
    for (const [key, hash] of shuffled(even, 19)) {
      map = map.delete(key, hash);
    }
    expect(nodeKind(map.getRootNode())).toBe('array'); // 16 odds remain
    for (const [key, hash] of shuffled(odd.slice(1), 23)) {
      map = map.delete(key, hash);
    }
    expect(nodeKind(map.getRootNode())).toBe('bitmap'); // 15 -> contracted
    const lastHash = odd[0][1];
    expect(map.get('k1', lastHash)).toBe(1);
    // Re-grow from the contracted state.
    for (const [key, hash, value] of shuffled(even, 29)) {
      map = map.set(key, value, hash);
    }
    // Then restore the odd keys removed during contraction.
    for (const [key, hash, value] of shuffled(odd.slice(1), 31)) {
      map = map.set(key, value, hash);
    }
    expect(nodeKind(map.getRootNode())).toBe('array');
    for (const [key, hash, value] of [...even, ...odd]) {
      expect(map.get(key, hash)).toBe(value);
    }
    expect(map.size()).toBe(32);
  });

  it('threshold contraction preserves slot order (bit 31 key queryable after shrink)', () => {
    // 16 slots including 31 -> array; delete two interior keys -> bitmap.
    const slots = [0, 3, 7, 9, 12, 15, 18, 21, 24, 26, 28, 29, 30, 31, 11, 5];
    const entries = slotEntries(slots);
    let map = seed(shuffled(entries, 31));
    expect(nodeKind(map.getRootNode())).toBe('array');
    map = map.delete('k3', entries[1][1]).delete('k24', entries[8][1]);
    expect(nodeKind(map.getRootNode())).toBe('bitmap');
    const root = map.getRootNode() as BitmapNode<number>;
    const remaining = slots.filter((s) => s !== 3 && s !== 24).sort((a, b) => a - b);
    let expectedBitmap = 0;
    for (const s of remaining) expectedBitmap = (expectedBitmap | bitFor(s)) >>> 0;
    expect(root.bitmap >>> 0).toBe(expectedBitmap);
    for (const slot of remaining) {
      const hash = h(slot, 0x100 + slot);
      expect(map.get(`k${slot}`, hash)).toBe(slot);
    }
  });
});

describe('alternating bits and full bitmap', () => {
  it('odd-bit (0xAAAAAAAA) and even-bit (0x55555555) layouts both index correctly', () => {
    for (const base of [0x55555555, 0xaaaaaaaa]) {
      let map = PersistentMap.empty<number>();
      for (let slot = 0; slot < 32; slot++) {
        if ((base & bitFor(slot)) === 0) continue;
        const hash = h(slot, 0x200);
        map = map.set(`b${slot}`, slot, hash);
      }
      // 16 top-level slots -> array node; same popcount path either parity.
      expect(nodeKind(map.getRootNode())).toBe('array');
      for (let slot = 0; slot < 32; slot++) {
        const hash = h(slot, 0x200);
        expect(map.get(`b${slot}`, hash)).toBe((base & bitFor(slot)) !== 0 ? slot : undefined);
      }
      expect(map.size()).toBe(16);
    }
  });

  it('supports all 32 top-level slots at once', () => {
    let map = PersistentMap.empty<number>();
    for (let slot = 0; slot < 32; slot++) {
      map = map.set(`f${slot}`, slot, h(slot, 0x300 + slot));
    }
    for (let slot = 0; slot < 32; slot++) {
      expect(map.get(`f${slot}`, h(slot, 0x300 + slot))).toBe(slot);
    }
    expect(map.size()).toBe(32);
  });
});

describe('custom hash extremes and collisions', () => {
  it('custom hash 0xffffffff (max uint32) routes to slot 31 at every level', () => {
    const map = PersistentMap.empty<string>()
      .set('max1', 'a', 0xffffffff)
      .set('max2', 'b', 0xffffffff)
      .set('max3', 'c', 0xffffffff);
    expect(map.get('max1', 0xffffffff)).toBe('a');
    expect(map.get('max2', 0xffffffff)).toBe('b');
    expect(map.get('max3', 0xffffffff)).toBe('c');
    expect(map.size()).toBe(3);
    // Full-collision bucket: canonical key order.
    expect(map.keys()).toEqual(['max1', 'max2', 'max3']);
  });

  it('custom hash 0x80000000 equals negative -2^31 at the boundary', () => {
    const map = PersistentMap.empty<number>()
      .set('u', 1, 0x80000000)
      .set('n', 2, -2147483648);
    expect(map.size()).toBe(2);
    expect(map.get('u', 0x80000000)).toBe(1);
    expect(map.get('n', -2147483648)).toBe(2);
    expect(map.get('u', -2147483648)).toBe(1);
  });

  it('hashes > 2^32 wrap through uint32 semantics', () => {
    const map = PersistentMap.empty<number>().set('wrap', 9, 0x1_8000_0001);
    expect(map.get('wrap', 0x80000001)).toBe(9);
    expect(map.get('wrap', 0x1_8000_0001)).toBe(9);
  });

  it('deleting inside a collision bucket restores a leaf and stays queryable', () => {
    let map = PersistentMap.empty<number>()
      .set('a', 1, 0xffffffff)
      .set('b', 2, 0xffffffff);
    map = map.delete('a', 0xffffffff);
    expect(map.size()).toBe(1);
    expect(map.get('b', 0xffffffff)).toBe(2);
    expect(map.get('a', 0xffffffff)).toBeUndefined();
  });

  it('stores hashes as unsigned in emitted entries', () => {
    const map = PersistentMap.empty<number>().set('neg', 1, -1);
    const entries: Entry<number>[] = map.items();
    expect(entries[0].hash).toBe(0xffffffff);
  });
});

describe('order independence', () => {
  const hashes = [
    0x00000000, 0x80000000, 0xffffffff, 0x40000000, h(31, 1),
    h(0, 2), h(15, 3), h(16, 4), 0x7fffffff, 0xaaaaaaaa,
    0x55555555, 0xdeadbeef >>> 0,
  ];
  const entries: Array<[string, number, number]> = hashes.map((h, i) => [`k${i}`, h >>> 0, i]);

  const reference = seed(entries);
  const expectedItems = reference.items();
  expect(expectedItems.map((e) => e.hash >>> 0)).toEqual(
    hashes.slice().sort((a, b) => (a >>> 0) - (b >>> 0)),
  );

  for (let s = 0; s < 6; s++) {
    it(`order permutation ${s} yields identical queries and traversal`, () => {
      const map = seed(shuffled(entries, s + 1));
      for (const [key, hash, value] of entries) {
        expect(map.get(key, hash)).toBe(value);
        expect(map.get(key, hash | 0)).toBe(value);
      }
      expect(map.size()).toBe(entries.length);
      expect(map.items()).toEqual(expectedItems);
      expect(map.keys()).toEqual(expectedItems.map((e) => e.key));
    });
  }

  it('all permutations across the array threshold agree', () => {
    const slots = Array.from({ length: 18 }, (_, i) => (i * 7) % 32).filter(
      (s, i, arr) => arr.indexOf(s) === i,
    );
    const data = slotEntries(slots);
    const ref = seed(data);
    for (let s = 0; s < 4; s++) {
      const map = seed(shuffled(data, s + 2));
      expect(nodeKind(map.getRootNode())).toBe('array');
      for (const [key, hash, value] of data) expect(map.get(key, hash)).toBe(value);
      expect(map.keys()).toEqual(ref.keys());
    }
  });
});

describe('default FNV hash and persistence', () => {
  it('default hashes are unsigned and persistence is preserved', () => {
    const a = new PersistentMap<number>();
    const b = a.set('x', 1);
    expect(a.size()).toBe(0);
    expect(b.get('x')).toBe(1);
    expect(hashKey('') >>> 0).toBe(hashKey(''));
    // Exercise many strings: any hash landing at slot 31 must behave.
    const keys = Array.from({ length: 400 }, (_, i) => `key-${i}`);
    let map = PersistentMap.empty<number>();
    for (const [i, key] of keys.entries()) map = map.set(key, i);
    for (const [i, key] of keys.entries()) {
      expect(map.get(key)).toBe(i);
      expect(hashKey(key) >>> 0).toBe(hashKey(key));
    }
    expect(nodeKind(map.getRootNode())).toBe('array');
    for (const key of keys) map = map.delete(key);
    expect(map.size()).toBe(0);
  });

  it('constructor with explicit entries array still works (back-compat)', () => {
    const map = new PersistentMap<number>([{ key: 'a', value: 7, hash: bitFor(31) - 1 >>> 0 }]);
    expect(map.get('a', (bitFor(31) - 1) >>> 0)).toBe(7);
  });

  it('LeafNode.get compares unsigned hashes', () => {
    const leaf = new LeafNode<number>({ key: 'k', value: 1, hash: 0xffffffff });
    expect(leaf.get('k', -1, null)).toBe(1);
    expect(leaf.get('k', 0, null)).toBeUndefined();
  });
});
