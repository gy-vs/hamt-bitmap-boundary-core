// Persistent HAMT (Hash Array Mapped Trie) core.
//
// Bitmap semantics are strictly unsigned 32-bit: every shift, complement and
// popcount input is normalized through `>>> 0` first. A 5-bit fragment yields
// slots 0..31, so bit 31 must never be interpreted as a sign bit.

export type Entry<V> = { key: string; value: V; hash: number };

/** First level consumes the top 5 bits; each subsequent level the next 5. */
const SHIFT_START = 27;
const SHIFT_STEP = 5;
/** Last level (shift null) consumes the final 2 bits. */
const SHIFT_FINAL = 2;
/** Bitmap grows into an array node at this many child subtrees; arrays shrink below it. */
const ARRAY_THRESHOLD = 16;

// --- uint32 primitives -------------------------------------------------

/** Coerce any JavaScript number hash into an unsigned 32-bit integer. */
export function toUint32(hash: number): number {
  return hash >>> 0;
}

/** 1 << slot under unsigned 32-bit semantics (safe for slot 31). */
export function bitFor(slot: number): number {
  return (1 << slot) >>> 0;
}

/** Extract the 5-bit fragment at `shift` (or the 2-bit terminal fragment). */
export function slotAt(hash: number, shift: number | null): number {
  return shift === null ? (hash >>> 0) & 3 : ((hash >>> shift) & 31) | 0;
}

export function popcount(bits: number): number {
  bits = bits >>> 0;
  bits = bits - ((bits >>> 1) & 0x55555555);
  bits = (bits & 0x33333333) + ((bits >>> 2) & 0x33333333);
  bits = (bits + (bits >>> 4)) & 0x0f0f0f0f;
  return ((bits >>> 24) + ((bits >>> 16) & 0xff) + ((bits >>> 8) & 0xff) + (bits & 0xff)) | 0;
}

/** Number of set bits strictly below `slot` — the compact-array index of slot. */
export function compactIndex(bitmap: number, slot: number): number {
  const below = (bitFor(slot) - 1) >>> 0;
  return popcount(((bitmap >>> 0) & below) >>> 0);
}

/** Inverse of compactIndex: slot addressed by compact-array index `idx`. */
export function slotForIndex(bitmap: number, idx: number): number {
  bitmap = bitmap >>> 0;
  let slot = 0;
  let remaining = idx;
  while (bitmap !== 0) {
    if ((bitmap & 1) !== 0) {
      if (remaining === 0) return slot;
      remaining--;
    }
    bitmap = bitmap >>> 1;
    slot++;
  }
  return -1;
}

function nextShift(shift: number | null): number | null {
  if (shift === null) return null;
  return shift === SHIFT_FINAL ? null : shift - SHIFT_STEP;
}

// --- nodes -------------------------------------------------------------

export abstract class Node<V> {
  abstract readonly count: number;
  abstract get(key: string, hash: number, shift: number | null): V | undefined;
  abstract set(leaf: LeafNode<V>, shift: number | null): Node<V>;
  abstract remove(key: string, hash: number, shift: number | null): Node<V> | null;
  abstract forEach(emit: (entry: Entry<V>) => void): void;
}

export class LeafNode<V> extends Node<V> {
  readonly count = 1;
  constructor(readonly entry: Entry<V>) {
    super();
  }
  get(key: string, hashRaw: number): V | undefined {
    const hash = hashRaw >>> 0;
    return hash === (this.entry.hash >>> 0) && this.entry.key === key
      ? this.entry.value
      : undefined;
  }
  set(leaf: LeafNode<V>, shift: number | null): Node<V> {
    if (
      leaf.entry.key === this.entry.key &&
      (leaf.entry.hash >>> 0) === (this.entry.hash >>> 0)
    ) {
      return leaf;
    }
    return mergeLeaves(this, leaf, shift);
  }
  remove(key: string, hashRaw: number): Node<V> | null {
    const hash = hashRaw >>> 0;
    return hash === (this.entry.hash >>> 0) && this.entry.key === key ? null : this;
  }
  forEach(emit: (entry: Entry<V>) => void): void {
    emit(this.entry);
  }
}

export class CollisionNode<V> extends Node<V> {
  readonly count: number;
  constructor(readonly hash: number, readonly entries: ReadonlyArray<Entry<V>>) {
    super();
    this.hash = hash >>> 0;
    this.count = entries.length;
  }
  get(key: string, hashRaw: number): V | undefined {
    const hash = hashRaw >>> 0;
    if (hash !== (this.hash >>> 0)) return undefined;
    return this.entries.find((entry) => entry.key === key)?.value;
  }
  set(leaf: LeafNode<V>): Node<V> {
    // Routing invariant: every entry here shares the full unsigned 32-bit hash.
    if ((leaf.entry.hash >>> 0) !== (this.hash >>> 0)) {
      throw new Error('collision node received a non-matching hash');
    }
    const idx = this.entries.findIndex((entry) => entry.key === leaf.entry.key);
    const entries =
      idx === -1
        ? insertSorted(this.entries, leaf.entry, compareEntryKey)
        : this.entries.map((entry, i) => (i === idx ? leaf.entry : entry));
    return new CollisionNode(this.hash, entries);
  }
  remove(key: string, hashRaw: number): Node<V> | null {
    const hash = hashRaw >>> 0;
    if (hash !== (this.hash >>> 0)) return this;
    if (!this.entries.some((entry) => entry.key === key)) return this;
    const entries = this.entries.filter((entry) => entry.key !== key);
    if (entries.length === 1) return new LeafNode(entries[0]);
    return new CollisionNode(this.hash, entries);
  }
  forEach(emit: (entry: Entry<V>) => void): void {
    // Identical hashes; key order is the canonical tiebreak.
    for (const entry of this.entries) emit(entry);
  }
}

export class BitmapNode<V> extends Node<V> {
  readonly bitmap: number;
  readonly count: number;
  constructor(bitmap: number, readonly children: ReadonlyArray<Node<V>>) {
    super();
    this.bitmap = bitmap >>> 0;
    let count = 0;
    for (const child of children) count += child.count;
    this.count = count;
  }
  get(key: string, hash: number, shift: number | null): V | undefined {
    const slot = slotAt(hash, shift);
    const bit = bitFor(slot);
    if (((this.bitmap >>> 0) & bit) === 0) return undefined;
    return this.children[compactIndex(this.bitmap, slot)].get(key, hash, nextShift(shift));
  }
  set(leaf: LeafNode<V>, shift: number | null): Node<V> {
    const hash = leaf.entry.hash >>> 0;
    const slot = slotAt(hash, shift);
    const bit = bitFor(slot);
    const idx = compactIndex(this.bitmap, slot);
    if (((this.bitmap >>> 0) & bit) === 0) {
      const bitmap = ((this.bitmap >>> 0) | bit) >>> 0;
      const children = [...this.children.slice(0, idx), leaf, ...this.children.slice(idx)];
      if (popcount(bitmap) >= ARRAY_THRESHOLD) return ArrayNode.fromSlots(bitmap, children);
      return new BitmapNode(bitmap, children);
    }
    const child = this.children[idx];
    const updated = child.set(leaf, nextShift(shift));
    if (updated === child) return this;
    const children = [...this.children.slice(0, idx), updated, ...this.children.slice(idx + 1)];
    return new BitmapNode(this.bitmap, children);
  }
  remove(key: string, hash: number, shift: number | null): Node<V> | null {
    const slot = slotAt(hash, shift);
    const bit = bitFor(slot);
    if (((this.bitmap >>> 0) & bit) === 0) return this;
    const idx = compactIndex(this.bitmap, slot);
    const updated = this.children[idx].remove(key, hash, nextShift(shift));
    if (updated === this.children[idx]) return this;
    if (updated === null) {
      const bitmap = ((this.bitmap >>> 0) & (~bit >>> 0)) >>> 0;
      if (bitmap === 0) return null;
      const children = [...this.children.slice(0, idx), ...this.children.slice(idx + 1)];
      return new BitmapNode(bitmap, children);
    }
    const children = [...this.children.slice(0, idx), updated, ...this.children.slice(idx + 1)];
    return new BitmapNode(this.bitmap, children);
  }
  forEach(emit: (entry: Entry<V>) => void): void {
    // Compact order == ascending set-bit (slot) order.
    for (const child of this.children) child.forEach(emit);
  }
}

export class ArrayNode<V> extends Node<V> {
  readonly count: number;
  constructor(readonly children: ReadonlyArray<Node<V> | null>) {
    super();
    let count = 0;
    for (const child of children) if (child) count += child.count;
    this.count = count;
  }
  /** Expand a bitmap node: packed children follow ascending slot order. */
  static fromSlots<V>(bitmap: number, packed: ReadonlyArray<Node<V>>): ArrayNode<V> {
    const slots: (Node<V> | null)[] = new Array(32).fill(null);
    let j = 0;
    for (let slot = 0; slot < 32; slot++) {
      if (((bitmap >>> 0) & bitFor(slot)) !== 0) slots[slot] = packed[j++];
    }
    return new ArrayNode(slots);
  }
  /** Contract back to a bitmap node: same ascending slot order, both directions. */
  private static toSlots<V>(
    children: ReadonlyArray<Node<V> | null>,
  ): { bitmap: number; packed: Node<V>[] } {
    let bitmap = 0;
    const packed: Node<V>[] = [];
    for (let slot = 0; slot < 32; slot++) {
      const child = children[slot];
      if (child !== null && child !== undefined) {
        bitmap = (bitmap | bitFor(slot)) >>> 0;
        packed.push(child);
      }
    }
    return { bitmap: bitmap >>> 0, packed };
  }
  get(key: string, hash: number, shift: number | null): V | undefined {
    return this.children[slotAt(hash, shift)]?.get(key, hash, nextShift(shift));
  }
  set(leaf: LeafNode<V>, shift: number | null): Node<V> {
    const slot = slotAt(leaf.entry.hash >>> 0, shift);
    const existing = this.children[slot];
    const updated =
      existing === null || existing === undefined ? leaf : existing.set(leaf, nextShift(shift));
    if (updated === existing) return this;
    const children = this.children.slice();
    children[slot] = updated;
    return new ArrayNode(children);
  }
  remove(key: string, hash: number, shift: number | null): Node<V> | null {
    const slot = slotAt(hash, shift);
    const existing = this.children[slot];
    if (existing === null || existing === undefined) return this;
    const updated = existing.remove(key, hash, nextShift(shift));
    if (updated === existing) return this;
    const children = this.children.slice();
    children[slot] = updated;
    // Threshold is measured in occupied child subtrees, not stored entries.
    const occupied = countOccupied(children);
    if (occupied === 0) return null;
    if (occupied < ARRAY_THRESHOLD) {
      const { bitmap, packed } = ArrayNode.toSlots(children);
      return new BitmapNode(bitmap, packed);
    }
    return new ArrayNode(children);
  }
  forEach(emit: (entry: Entry<V>) => void): void {
    for (let slot = 0; slot < 32; slot++) {
      const child = this.children[slot];
      if (child !== null && child !== undefined) child.forEach(emit);
    }
  }
}

function countOccupied<V>(children: ReadonlyArray<Node<V> | null>): number {
  let n = 0;
  for (const child of children) if (child !== null && child !== undefined) n++;
  return n;
}

function mergeLeaves<V>(
  a: LeafNode<V>,
  b: LeafNode<V>,
  shift: number | null,
): Node<V> {
  const ha = a.entry.hash >>> 0;
  const hb = b.entry.hash >>> 0;
  const slotA = slotAt(ha, shift);
  const slotB = slotAt(hb, shift);
  if (slotA === slotB) {
    if (shift === null) {
      // Same terminal 2-bit bucket: only possible for truly identical hashes.
      if (ha === hb) {
        return new CollisionNode(ha, [a.entry, b.entry].sort(compareEntryKey));
      }
      throw new Error('distinct hashes merged into one terminal slot');
    }
    return new BitmapNode(bitFor(slotA), [mergeLeaves(a, b, nextShift(shift))]);
  }
  const [lo, hi] = slotA < slotB ? [slotA, slotB] : [slotB, slotA];
  const [childLo, childHi] = slotA < slotB ? [a, b] : [b, a];
  // Terminal-level bitmap only uses slots 0..3.
  const bitmap = (bitFor(lo) | bitFor(hi)) >>> 0;
  return new BitmapNode(bitmap, [childLo, childHi]);
}

function compareEntryKey<V>(a: Entry<V>, b: Entry<V>): number {
  return a.key < b.key ? -1 : a.key > b.key ? 1 : 0;
}

function insertSorted<V>(
  entries: ReadonlyArray<Entry<V>>,
  entry: Entry<V>,
  compare: (a: Entry<V>, b: Entry<V>) => number,
): Entry<V>[] {
  let lo = 0;
  let hi = entries.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (compare(entries[mid], entry) < 0) lo = mid + 1;
    else hi = mid;
  }
  return [...entries.slice(0, lo), entry, ...entries.slice(lo)];
}

// --- public map --------------------------------------------------------

export function hashKey(value: string): number {
  let hash = 2166136261;
  for (const char of value) {
    hash = Math.imul(hash ^ char.charCodeAt(0), 16777619) >>> 0;
  }
  return hash >>> 0;
}

export type NodeKind = 'empty' | 'leaf' | 'bitmap' | 'array' | 'collision';

export function nodeKind(node: Node<unknown> | null | undefined): NodeKind {
  if (node === null || node === undefined) return 'empty';
  if (node instanceof LeafNode) return 'leaf';
  if (node instanceof ArrayNode) return 'array';
  if (node instanceof CollisionNode) return 'collision';
  return 'bitmap';
}

export class PersistentMap<V> {
  private readonly root: Node<V> | null;

  /** Empty map, or one pre-seeded with explicit entries (back-compat). */
  constructor(entries?: ReadonlyArray<Entry<V>>);
  /** Internal: wrap a root node. */
  constructor(root: Node<V> | null);
  constructor(rootOrEntries?: Node<V> | ReadonlyArray<Entry<V>> | null) {
    if (Array.isArray(rootOrEntries)) {
      let root: Node<V> | null = null;
      for (const raw of rootOrEntries) {
        const leaf = new LeafNode<V>({ key: raw.key, value: raw.value, hash: raw.hash >>> 0 });
        root = root === null ? leaf : root.set(leaf, SHIFT_START);
      }
      this.root = root;
    } else {
      this.root = (rootOrEntries as Node<V> | null | undefined) ?? null;
    }
  }

  static empty<V>(): PersistentMap<V> {
    return new PersistentMap<V>(null);
  }

  getRootNode(): Node<V> | null {
    return this.root;
  }

  get(key: string, hash: number = hashKey(key)): V | undefined {
    return this.root?.get(key, hash >>> 0, SHIFT_START);
  }

  set(key: string, value: V, hash: number = hashKey(key)): PersistentMap<V> {
    const leaf = new LeafNode<V>({ key, value, hash: hash >>> 0 });
    const root = this.root === null ? leaf : this.root.set(leaf, SHIFT_START);
    return new PersistentMap<V>(root);
  }

  delete(key: string, hash: number = hashKey(key)): PersistentMap<V> {
    if (this.root === null) return this;
    const root = this.root.remove(key, hash >>> 0, SHIFT_START);
    return root === this.root ? this : new PersistentMap<V>(root);
  }

  size(): number {
    return this.root === null ? 0 : this.root.count;
  }

  /** Canonical traversal: ascending unsigned hash, then key on collisions. */
  items(): Entry<V>[] {
    const out: Entry<V>[] = [];
    this.root?.forEach((entry) => out.push(entry));
    return out;
  }

  keys(): string[] {
    return this.items().map((entry) => entry.key);
  }

  values(): V[] {
    return this.items().map((entry) => entry.value);
  }
}
