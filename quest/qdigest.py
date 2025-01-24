import math
import time
from typing import List, Tuple

class QuantileDigest:
    MAX_BITS = 64
    INITIAL_CAPACITY = 1

    class TraversalOrder:
        FORWARD = 0
        REVERSE = 1

    def __init__(self, max_error: float):
        assert 0 <= max_error <= 1, "maxError must be in range [0, 1]"

        self.max_error = max_error
        self.weighted_count = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.root = -1
        self.next_node = 0
        self.counts = [0.0] * self.INITIAL_CAPACITY
        self.levels = [0] * self.INITIAL_CAPACITY
        self.values = [0] * self.INITIAL_CAPACITY
        self.lefts = [-1] * self.INITIAL_CAPACITY
        self.rights = [-1] * self.INITIAL_CAPACITY
        self.free_count = 0
        self.first_free = -1

    def add(self, value: int, weight: float = 1.0):
        assert weight > 0, "weight must be > 0"

        needs_compression = False
        
        self.max = max(self.max, value)
        self.min = min(self.min, value)

        previous_count = self.weighted_count
        self._insert(value, weight)

        compression_factor = self._calculate_compression_factor()
        if needs_compression or int(previous_count) / compression_factor != int(self.weighted_count) / compression_factor:
            self.compress()

    def merge(self, other: 'QuantileDigest'):
        self.root = self._merge(self.root, other, other.root)
        self.max = max(self.max, other.max)
        self.min = min(self.min, other.min)
        self.compress()

    def get_quantiles_lower_bound(self, quantiles: List[float]) -> List[int]:
        assert all(quantiles[i] <= quantiles[i+1] for i in range(len(quantiles)-1)), "quantiles must be sorted in increasing order"
        assert all(0 <= q and q <= 1 for q in quantiles), "quantile must be between [0,1]"

        reversed_quantiles = list(reversed(quantiles))
        result = []
        iterator = iter(reversed_quantiles)
        current_quantile = next(iterator, None)

        def callback(node):
            nonlocal current_quantile
            self.sum += self.counts[node]

            while current_quantile is not None and self.sum > (1.0 - current_quantile) * self.weighted_count:
                value = max(self._lower_bound(node), self.min)
                result.append(value)
                current_quantile = next(iterator, None)

            return current_quantile is not None

        self.sum = 0
        self._post_order_traversal(self.root, callback, self.TraversalOrder.REVERSE)

        # Handle remaining quantiles
        while current_quantile is not None:
            result.append(self.min)
            current_quantile = next(iterator, None)

        return list(reversed(result))
    
    def get_quantiles_upper_bound(self, quantiles: List[float]) -> List[int]:
        assert all(quantiles[i] <= quantiles[i+1] for i in range(len(quantiles)-1)), "quantiles must be sorted in increasing order"
        assert all(0 <= q and q <= 1 for q in quantiles), "quantile must be between [0,1]"

        result = []
        iterator = iter(quantiles)
        current_quantile = next(iterator, None)

        def callback(node):
            nonlocal current_quantile
            self.sum += self.counts[node]

            while current_quantile is not None and self.sum > current_quantile * self.weighted_count:
                value = min(self._upper_bound(node), self.max)
                result.append(value)
                current_quantile = next(iterator, None)

            return current_quantile is not None

        self.sum = 0
        self._post_order_traversal(self.root, callback)

        # Handle remaining quantiles
        while current_quantile is not None:
            result.append(self.max)
            current_quantile = next(iterator, None)

        return result

    def _post_order_traversal(self, node, callback, order = TraversalOrder.FORWARD):
        if node == -1:
            return False

        if order == self.TraversalOrder.FORWARD:
            if self.lefts[node]!= -1 and not self._post_order_traversal(self.lefts[node], callback, order):
                return False
            if self.rights[node]!= -1 and not self._post_order_traversal(self.rights[node], callback, order):
                return False
        else:
            if self.rights[node]!= -1 and not self._post_order_traversal(self.rights[node], callback, order):
                return False
            if self.lefts[node]!= -1 and not self._post_order_traversal(self.lefts[node], callback, order):
                return False

        return callback(node)

    def _lower_bound(self, node):
        level = self.levels[node]
        value = self.values[node]
        if level > 0:
            return self._get_bit_representation(value) & (~(0xFFFF_FFFF_FFFF_FFFF >> (self.MAX_BITS - level)))
        return value
    
    def _upper_bound(self, node):
        level = self.levels[node]
        value = self.values[node]
        if level > 0:
            return self._get_bit_representation(value) | (0xFFFF_FFFF_FFFF_FFFF >> (self.MAX_BITS - level))
        return value

    # def get_quantiles(self, quantiles: List[float]) -> List[int]:
    #     assert all(0 <= q <= 1 for q in quantiles), "quantiles must be in range [0, 1]"
    #     assert sorted(quantiles) == quantiles, "quantiles must be sorted in increasing order"

    #     result = []
    #     if self.weighted_count == 0:
    #         return [0] * len(quantiles)

    #     current_node = self.root
    #     current_weight = 0
    #     current_level = self._max_level()

    #     for q in quantiles:
    #         target_weight = q * self.weighted_count

    #         while True:
    #             if current_node == -1:
    #                 result.append(self._bits_to_long(current_level, 0))
    #                 break

    #             left_weight = current_weight
    #             right_weight = current_weight + self.counts[current_node]

    #             if target_weight < left_weight + (right_weight - left_weight) / 2:
    #                 if self.lefts[current_node] == -1:
    #                     result.append(self._bits_to_long(current_level, 0))
    #                     break
    #                 current_node = self.lefts[current_node]
    #             else:
    #                 if self.rights[current_node] == -1:
    #                     result.append(self.values[current_node])
    #                     break
    #                 current_weight = right_weight
    #                 current_node = self.rights[current_node]
    #             current_level -= 1

    #     return result

    def get_quantiles(self, quantiles: List[float]) -> List[int]:
        return self.get_quantiles_upper_bound(quantiles)

    def _insert(self, value, count):
        last_branch = 0
        parent = -1
        current = self.root

        while True:
            if current == -1:
                self._set_child(parent, last_branch, self._create_leaf(value, count))
                return

            current_value = self.values[current]
            current_level = self.levels[current]
            if not self._in_same_subtree(value, current_value, current_level):
                # if value and node.value are not in the same branch given node's level,
                # insert a parent above them at the point at which branches diverge
                self._set_child(parent, last_branch, self._make_siblings(current, self._create_leaf(value, count)))
                return

            if current_level == 0 and current_value == value:
                # found the node
                self.counts[current] += count
                self.weighted_count += count
                return

            # we're on the correct branch of the tree and we haven't reached a leaf, so keep going down
            # bit shift to handle negative number representation in python
            branch = self._get_bit_representation(value) & self._get_branch_mask(current_level)
            
            parent = current
            last_branch = branch

            if branch == 0:
                current = self.lefts[current]
            else:
                current = self.rights[current]

    def _eager_insert(self, value, count):
        last_branch = 0
        parent = -1
        current = self.root

        while True:
            if current == -1:
                self._set_child(parent, last_branch, self._create_leaf(value, count))
                return

            current_value = self.values[current]
            current_level = self.levels[current]
            if not self._in_same_subtree(value, current_value, current_level):
                # if value and node.value are not in the same branch given node's level,
                # insert a parent above them at the point at which branches diverge
                self._set_child(parent, last_branch, self._make_siblings(current, self._create_leaf(value, count)))
                return

            # convert to eager adding of nodes to tree
            if current_level == 0 and current_value == value:
                # found the node
                self.counts[current] += count
                self.weighted_count += count
                return

            # we're on the correct branch of the tree and we haven't reached a leaf, so keep going down
            # bit shift to handle negative number representation in python
            branch = self._get_bit_representation(value) & self._get_branch_mask(current_level)
            
            parent = current
            last_branch = branch

            if branch == 0:
                current = self.lefts[current]
            else:
                current = self.rights[current]

    def _set_child(self, parent, branch, child):
        if parent == -1:
            self.root = child
        elif branch == 0:
            self.lefts[parent] = child
        else:
            self.rights[parent] = child

    def _make_siblings(self, first, second):
        first_value = self.values[first]
        second_value = self.values[second]

        parent_level = (self._get_bit_representation(first_value) ^ self._get_bit_representation(second_value)).bit_length()
        parent = self._create_node(first_value, parent_level, 0)

        # the branch is given by the bit at the level one below parent
        branch = self._get_bit_representation(first_value)& self._get_branch_mask(self.levels[parent])

        if branch == 0:
            self.lefts[parent] = first
            self.rights[parent] = second
        else:
            self.lefts[parent] = second
            self.rights[parent] = first

        return parent

    def _create_leaf(self, value, count):
        return self._create_node(value, 0, count)
    
    def _create_node(self, value, level, count):
        node = self._pop_free()

        if node == -1:
            if self.next_node == len(self.counts):
                # try to double the array, but don't allocate too much to avoid going over the upper bound of nodes
                # by a large margin (hence, the heuristic to not allocate more than k / 5 nodes)
                new_size = len(self.counts) + int(min(len(self.counts), self._calculate_compression_factor() / 5 + 1))
                self.counts = self.counts[:new_size] + [0]*(new_size - len(self.counts))
                self.levels = self.levels[:new_size] + [0]*(new_size - len(self.levels))
                self.values = self.values[:new_size] + [0]*(new_size - len(self.values))
                self.lefts = self.lefts[:new_size] + [0]*(new_size - len(self.lefts))
                self.rights = self.rights[:new_size] + [0]*(new_size - len(self.rights))

            node = self.next_node
            self.next_node += 1

        self.weighted_count += count
        self.values[node] = value
        self.levels[node] = level
        self.counts[node] = count

        self.lefts[node] = -1
        self.rights[node] = -1

        return node

    def _copy_recursive(self, other, other_node):
        if other_node == -1:
            return other_node

        node = self._create_node(other.values[other_node], other.levels[other_node], other.counts[other_node])

        if other.lefts[other_node] != -1:
            # variable needed because the array may be re-allocated during merge()
            left = self._copy_recursive(other, other.lefts[other_node])
            self.lefts[node] = left

        if other.rights[other_node] != -1:
            # variable needed because the array may be re-allocated during merge()
            right = self._copy_recursive(other, other.rights[other_node])
            self.rights[node] = right

        return node

    def try_remove(self, node):
        assert node != -1, "node is -1"

        left = self.lefts[node]
        right = self.rights[node]

        if left == -1 and right == -1:
            # leaf, just remove it
            self._remove(node)
            return -1

        if left != -1 and right != -1:
            # node has both children so we can't physically remove it
            self.counts[node] = 0
            return node

        # node has a single child, so remove it and return the child
        self._remove(node)
        if left != -1:
            return left
        else:
            return right
        
    def _remove(self, node):
        if node == self.next_node - 1:
            # if we're removing the last node, no need to add it to the free list
            self.next_node -= 1
        else:
            self._push_free(node)

        if node == self.root:
            self.root = -1

    def _pop_free(self):
        node = self.first_free

        if node == -1:
            return node

        self.first_free = self.lefts[self.first_free]
        self.free_count -= 1

        return node

    def _push_free(self, node):
        self.lefts[node] = self.first_free
        self.first_free = node
        self.free_count += 1

    def compress(self):
        bound = math.floor(self.weighted_count / self._calculate_compression_factor())

        def compress_callback(node):
            # if children's weights are 0 remove them and shift the weight to their parent
            left = self.lefts[node]
            right = self.rights[node]

            if left == -1 and right == -1:
                # leaf, nothing to do
                return True

            left_count = 0.0 if left == -1 else self.counts[left]
            right_count = 0.0 if right == -1 else self.counts[right]

            should_compress = (self.counts[node] + left_count + right_count) < bound

            if left != -1 and should_compress:
                self.lefts[node] = self.try_remove(left)
                self.counts[node] += left_count

            if right != -1 and should_compress:
                self.rights[node] = self.try_remove(right)
                self.counts[node] += right_count

            return True

        self._post_order_traversal(self.root, compress_callback)


    def _merge(self, node, other, other_node):
        if other_node == -1:
            return node
        elif node == -1:
            return self._copy_recursive(other, other_node)
        elif not self._in_same_subtree(self.values[node], other.values[other_node], max(self.levels[node], other.levels[other_node])):
            return self._make_siblings(node, self._copy_recursive(other, other_node))
        elif self.levels[node] > other.levels[other_node]:
            branch = self._get_bit_representation(other.values[other_node]) & self._get_branch_mask(self.levels[node])

            if branch == 0:
                # variable needed because the array may be re-allocated during merge()
                left = self._merge(self.lefts[node], other, other_node)
                self.lefts[node] = left
            else:
                # variable needed because the array may be re-allocated during merge()
                right = self._merge(self.rights[node], other, other_node)
                self.rights[node] = right
            return node
        elif self.levels[node] < other.levels[other_node]:
            branch = self._get_bit_representation(self.values[node]) & self._get_branch_mask(other.levels[other_node])

            # variables needed because the arrays may be re-allocated during merge()
            if branch == 0:
                left = self._merge(node, other, other.lefts[other_node])
                right = self._copy_recursive(other, other.rights[other_node])
            else:
                left = self._copy_recursive(other, other.lefts[other_node])
                right = self._merge(node, other, other.rights[other_node])

            result = self._create_node(other.values[other_node], other.levels[other_node], other.counts[other_node])
            self.lefts[result] = left
            self.rights[result] = right

            return result

        # else, they must be at the same level and on the same path, so just bump the counts
        self.weighted_count += other.counts[other_node]
        self.counts[node] += other.counts[other_node]

        # variables needed because the arrays may be re-allocated during merge()
        left = self._merge(self.lefts[node], other, other.lefts[other_node])
        right = self._merge(self.rights[node], other, other.rights[other_node])
        self.lefts[node] = left
        self.rights[node] = right

        return node
    
    def _in_same_subtree(self, value1, value2, level):
        return level == self.MAX_BITS or (self._get_bit_representation(value1) >> level) == (self._get_bit_representation(value2) >> level)
    
    def _compute_max_path_weight(self, node):
        if node == -1 or self.levels[node] == 0:
            return 0

        left_max_weight = self._compute_max_path_weight(self.lefts[node])
        right_max_weight = self._compute_max_path_weight(self.rights[node])

        return max(left_max_weight, right_max_weight) + self.counts[node]
    
    def get_confidence_factor(self) -> float:
        return self._compute_max_path_weight(self.root) / self.weighted_count
    
    def equivalent(self, other):
        return self.get_node_count() == other.get_node_count() \
            and self.min == other.min \
            and self.max == other.max \
            and self.weighted_count == other.weighted_count
    
    def validate(self):
        sum_value = 0.0
        node_count = 0

        free_slots = self._compute_free_list()
        assert len(free_slots) == self.free_count, f"Free count ({self.free_count}) doesn't match actual free slots: {len(free_slots)}"

        if self.root != -1:
            self._validate_structure(self.root, free_slots)

            def callback(node):
                nonlocal sum_value, node_count
                sum_value += self.counts[node]
                node_count += 1
                return True

            self._post_order_traversal(self.root, callback)
        assert node_count == self.get_node_count(), \
            f"Actual node count ({node_count}) doesn't match summary ({self.get_node_count()})"

    def _compute_free_list(self):
        free_slots = set()
        index = self.first_free
        while index != -1:
            free_slots.add(index)
            index = self.lefts[index]
        return free_slots
    
    def _validate_structure(self, node, free_nodes):
        assert self.levels[node] >= 0, "Node level must be non-negative"

        assert node not in free_nodes, f"Node is in list of free slots: {node}"
        
        if self.lefts[node] != -1:
            self._validate_branch_structure(node, self.lefts[node], self.rights[node], True)
            self._validate_structure(self.lefts[node], free_nodes)

        if self.rights[node] != -1:
            self._validate_branch_structure(node, self.rights[node], self.lefts[node], False)
            self._validate_structure(self.rights[node], free_nodes)
    
    def _validate_branch_structure(self, parent, child, other_child, is_left):
        assert self.levels[child] < self.levels[parent], f"Child level ({self.levels[child]}) should be smaller than parent level ({self.levels[parent]})"

        branch = self._get_bit_representation(self.values[child]) & (1 << (self.levels[parent] - 1))
        assert (branch == 0 and is_left) or (branch != 0 and not is_left), "Value of child node is inconsistent with its branch"

        assert self.counts[parent] > 0 or self.counts[child] > 0 or other_child != -1, "Found a linear chain of zero-weight nodes"

    def _calculate_compression_factor(self) -> int:
        if self.root == -1:
            return 1
        return max(int((self.levels[self.root]+1)/ self.max_error),1)

    def _max_level(self) -> int:
        return self.MAX_BITS - (self.min ^ self.max).bit_length() + 1
    
    def get_count(self) -> int:
        return self.weighted_count
    
    def get_node_count(self) -> int:
        return self.next_node - self.free_count
    
    def _get_bit_representation(self, value: int) -> int:
        return ((1 << self.MAX_BITS) - 1) & value
    
    def get_min(self):
        chosen = self.min

        def callback(node):
            return True

        self._post_order_traversal(self.root, callback, self.TraversalOrder.FORWARD)

        return max(self.min, chosen)
    
    def get_max(self):
        chosen = self.max

        def callback(node):
            return True

        self._post_order_traversal(self.root, callback, self.TraversalOrder.REVERSE)

        return min(self.max, chosen)

    @staticmethod
    def _bits_to_long(level: int, bits: int) -> int:
        return bits << (64 - level)

    @staticmethod
    def _get_branch_mask(level: int) -> int:
        return 1 << (level - 1)
    
    
