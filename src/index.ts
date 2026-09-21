export type Entry<V> = { key: string; value: V; hash: number };

const BITS = 5;
const WIDTH = 1 << BITS; // 32 slots per node
const SLOT_MASK = WIDTH - 1;
const EXPAND_THRESHOLD = 16; // bitmap node -> array node above this many children
const SHRINK_THRESHOLD = 8; // array node -> bitmap node at or below this many children

type Node<V> = BitmapNode<V> | ArrayNode<V> | CollisionNode<V>;

type BitmapNode<V> = {
  type: 'bitmap';
  bitmap: number; // uint32, bit i set => slot i occupied
  children: (Node<V> | Entry<V>)[]; // compact, ascending slot order
};

type ArrayNode<V> = {
  type: 'array';
  count: number;
  children: (Node<V> | Entry<V> | undefined)[]; // length WIDTH, index === slot
};

type CollisionNode<V> = {
  type: 'collision';
  hash: number; // uint32, shared by every entry
  entries: Entry<V>[]; // sorted by key so traversal order is canonical
};

// --- unsigned 32-bit helpers: every shift/complement/popcount input is uint32 ---

const uint32 = (value: number) => value >>> 0;

const bitOf = (slot: number) => (1 << slot) >>> 0; // slot 31 => 0x80000000, never negative

const maskBelow = (slot: number) => (bitOf(slot) - 1) >>> 0; // bits strictly below slot

function popcount(value: number): number {
  let x = value >>> 0;
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return (x * 0x01010101) >>> 24;
}

// compact-array index of a slot: number of occupied bits below its bit
const indexOf = (bitmap: number, slot: number) => popcount(uint32(bitmap) & maskBelow(slot));

const fragment = (hash: number, shift: number) => (uint32(hash) >>> shift) & SLOT_MASK;

const isEntry = <V>(node: Node<V> | Entry<V>): node is Entry<V> =>
  !('type' in node);

const bitmapNode = <V>(bitmap: number, children: (Node<V> | Entry<V>)[]): BitmapNode<V> => ({
  type: 'bitmap',
  bitmap: uint32(bitmap),
  children,
});

function collisionNode<V>(hash: number, entries: Entry<V>[]): CollisionNode<V> {
  return { type: 'collision', hash: uint32(hash), entries: entries.slice().sort(byKey) };
}

const byKey = <V>(a: Entry<V>, b: Entry<V>) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0);

// bitmap node -> array node: child of slot s moves from compact index to index s,
// iterating slots ascending keeps the same slot order on both sides of the threshold
function expand<V>(node: BitmapNode<V>): ArrayNode<V> {
  const children: (Node<V> | Entry<V> | undefined)[] = new Array(WIDTH).fill(undefined);
  let index = 0;
  for (let slot = 0; slot < WIDTH; slot++) {
    if ((node.bitmap & bitOf(slot)) >>> 0 !== 0) children[slot] = node.children[index++];
  }
  return { type: 'array', count: node.children.length, children };
}

// array node -> bitmap node: occupied slots ascending become the compact children
function shrink<V>(node: ArrayNode<V>): BitmapNode<V> {
  let bitmap = 0;
  const children: (Node<V> | Entry<V>)[] = [];
  for (let slot = 0; slot < WIDTH; slot++) {
    const child = node.children[slot];
    if (child !== undefined) {
      bitmap = (bitmap | bitOf(slot)) >>> 0;
      children.push(child);
    }
  }
  return bitmapNode(bitmap, children);
}

function insert<V>(
  current: Node<V> | Entry<V> | undefined,
  shift: number,
  entry: Entry<V>,
): { out: Node<V> | Entry<V>; added: boolean } {
  if (current === undefined) return { out: entry, added: true };

  if (isEntry(current)) {
    if (current.hash === entry.hash) {
      if (current.key === entry.key) return { out: entry, added: false };
      return { out: collisionNode(entry.hash, [current, entry]), added: true };
    }
    // hashes differ: split with a bitmap node holding the existing leaf, then descend
    const split = bitmapNode(bitOf(fragment(current.hash, shift)), [current]);
    return insert(split, shift, entry);
  }

  if (current.type === 'collision') {
    if (current.hash === entry.hash) {
      const entries = current.entries.slice();
      const at = entries.findIndex(item => item.key >= entry.key);
      if (at >= 0 && entries[at].key === entry.key) {
        entries[at] = entry;
        return { out: { ...current, entries }, added: false };
      }
      entries.splice(at < 0 ? entries.length : at, 0, entry);
      return { out: { ...current, entries }, added: true };
    }
    const split = bitmapNode(bitOf(fragment(current.hash, shift)), [current]);
    return insert(split, shift, entry);
  }

  const slot = fragment(entry.hash, shift);

  if (current.type === 'bitmap') {
    const bit = bitOf(slot);
    const index = indexOf(current.bitmap, slot);
    if ((current.bitmap & bit) >>> 0 !== 0) {
      const result = insert(current.children[index], shift + BITS, entry);
      const children = current.children.slice();
      children[index] = result.out;
      return { out: { ...current, children }, added: result.added };
    }
    const bitmap = (current.bitmap | bit) >>> 0;
    const children = [
      ...current.children.slice(0, index),
      entry,
      ...current.children.slice(index),
    ];
    const next = bitmapNode(bitmap, children);
    return { out: children.length > EXPAND_THRESHOLD ? expand(next) : next, added: true };
  }

  const child = current.children[slot];
  const result = insert(child, shift + BITS, entry);
  const children = current.children.slice();
  children[slot] = result.out;
  const count = child === undefined ? current.count + 1 : current.count;
  return { out: { ...current, children, count }, added: result.added };
}

function remove<V>(
  current: Node<V> | Entry<V>,
  shift: number,
  hash: number,
  key: string,
): { out: Node<V> | Entry<V> | undefined; removed: boolean } {
  if (isEntry(current)) {
    return current.hash === hash && current.key === key
      ? { out: undefined, removed: true }
      : { out: current, removed: false };
  }

  if (current.type === 'collision') {
    if (current.hash !== hash) return { out: current, removed: false };
    const at = current.entries.findIndex(item => item.key === key);
    if (at < 0) return { out: current, removed: false };
    const entries = [...current.entries.slice(0, at), ...current.entries.slice(at + 1)];
    if (entries.length === 1) return { out: entries[0], removed: true };
    return { out: { ...current, entries }, removed: true };
  }

  const slot = fragment(hash, shift);

  if (current.type === 'bitmap') {
    const bit = bitOf(slot);
    if ((current.bitmap & bit) >>> 0 === 0) return { out: current, removed: false };
    const index = indexOf(current.bitmap, slot);
    const result = remove(current.children[index], shift + BITS, hash, key);
    if (!result.removed) return { out: current, removed: false };
    if (result.out === undefined) {
      const bitmap = (current.bitmap & ~bit) >>> 0; // complement on uint32 semantics
      if (bitmap === 0) return { out: undefined, removed: true };
      const children = [
        ...current.children.slice(0, index),
        ...current.children.slice(index + 1),
      ];
      return { out: bitmapNode(bitmap, children), removed: true };
    }
    const children = current.children.slice();
    children[index] = result.out;
    return { out: { ...current, children }, removed: true };
  }

  const child = current.children[slot];
  if (child === undefined) return { out: current, removed: false };
  const result = remove(child, shift + BITS, hash, key);
  if (!result.removed) return { out: current, removed: false };
  const children = current.children.slice();
  children[slot] = result.out;
  if (result.out === undefined) {
    const count = current.count - 1;
    if (count <= SHRINK_THRESHOLD) return { out: shrink({ ...current, count, children }), removed: true };
    return { out: { ...current, count, children }, removed: true };
  }
  return { out: { ...current, children }, removed: true };
}

function* walk<V>(node: Node<V> | Entry<V> | undefined): Generator<Entry<V>> {
  if (node === undefined) return;
  if (isEntry(node)) {
    yield node;
    return;
  }
  if (node.type === 'collision') {
    yield* node.entries;
    return;
  }
  if (node.type === 'bitmap') {
    for (const child of node.children) yield* walk(child);
    return;
  }
  for (const child of node.children) if (child !== undefined) yield* walk(child);
}

export class PersistentMap<V> {
  private root: Node<V> | Entry<V> | undefined;
  private length: number;

  constructor(entries: Entry<V>[] = []) {
    let root: Node<V> | Entry<V> | undefined;
    let length = 0;
    for (const item of entries) {
      const result = insert(root, 0, { ...item, hash: uint32(item.hash) });
      root = result.out;
      length += result.added ? 1 : 0;
    }
    this.root = root;
    this.length = length;
  }

  private static from<V>(root: Node<V> | Entry<V> | undefined, length: number): PersistentMap<V> {
    const map = new PersistentMap<V>();
    map.root = root;
    map.length = length;
    return map;
  }

  get(key: string, hash = hashKey(key)): V | undefined {
    const target = uint32(hash);
    let node = this.root;
    let shift = 0;
    while (node !== undefined) {
      if (isEntry(node)) return node.hash === target && node.key === key ? node.value : undefined;
      if (node.type === 'collision') {
        if (node.hash !== target) return undefined;
        return node.entries.find(item => item.key === key)?.value;
      }
      const slot = fragment(target, shift);
      if (node.type === 'bitmap') {
        const bit = bitOf(slot);
        if ((node.bitmap & bit) >>> 0 === 0) return undefined;
        node = node.children[indexOf(node.bitmap, slot)];
      } else {
        node = node.children[slot];
      }
      shift += BITS;
    }
    return undefined;
  }

  set(key: string, value: V, hash = hashKey(key)): PersistentMap<V> {
    const entry: Entry<V> = { key, value, hash: uint32(hash) };
    const result = insert(this.root, 0, entry);
    return PersistentMap.from(result.out, this.length + (result.added ? 1 : 0));
  }

  delete(key: string, hash = hashKey(key)): PersistentMap<V> {
    if (this.root === undefined) return this;
    const result = remove(this.root, 0, uint32(hash), key);
    if (!result.removed) return this;
    return PersistentMap.from(result.out, this.length - 1);
  }

  size(): number {
    return this.length;
  }

  items(): Entry<V>[] {
    return [...walk(this.root)];
  }
}

export function hashKey(value: string): number {
  let hash = 2166136261;
  for (const char of value) hash = Math.imul(hash ^ char.charCodeAt(0), 16777619);
  return hash >>> 0;
}
