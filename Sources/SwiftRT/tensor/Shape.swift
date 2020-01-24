//******************************************************************************
// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
import Foundation

//==============================================================================
//
public protocol ShapeArrayProtocol:
    RandomAccessCollection,
    MutableCollection,
    CustomStringConvertible,
    Equatable,
    Codable
    where Element == Int, Index == Int
{
    // types
    associatedtype Storage

    // properties
    var array: [Int] { get }
    var storage: Storage { get set }

    // initialzers
    init(_ data: Storage)
    init?(_ data: Storage?)
}

//==============================================================================
//
public struct ShapeArray<Storage> : ShapeArrayProtocol {
    /// the collection as a Swift Array
    @inlinable
    public var array: [Int] { [Int](self) }
    /// some value object used for storage space
    public var storage: Storage
    /// alias
    @inlinable
    public var tuple: Storage { storage }
    /// the number of elements in the array
    @inlinable
    public var count: Int {
        MemoryLayout<Storage>.size / MemoryLayout<Element>.size
    }
    /// starting index
    @inlinable
    public var startIndex: Int { 0 }
    /// ending index
    @inlinable
    public var endIndex: Int { count }

    /// description
    public var description: String { String(describing: array) }
    
    //--------------------------------------------------------------------------
    // initializers
    @inlinable
    public init(_ data: Storage) {
        assert(MemoryLayout<Storage>.size % MemoryLayout<Int>.size == 0,
               "Storage size must be multiple of Int size")
        storage = data
    }

    @inlinable
    public init?(_ data: Storage?) {
        guard let data = data else { return nil }
        self.init(data)
    }

//    @inlinable
//    public init() {
//        assert(MemoryLayout<Storage>.size % MemoryLayout<Int>.size == 0,
//               "Storage size must be multiple of Int size")
//        memset(&storage, 0, MemoryLayout<Storage>.size)
//    }

    //--------------------------------------------------------------------------
    // Equatable
    @inlinable
    public static func == (lhs: Self, rhs: Self) -> Bool {
        withUnsafeBytes(of: lhs.storage) { lhsPtr in
            withUnsafeBytes(of: rhs.storage) { rhsPtr in
                memcmp(lhsPtr.baseAddress!,
                       rhsPtr.baseAddress!,
                       MemoryLayout<Storage>.size) == 0
            }
        }
    }

    @inlinable
    public static func == (lhs: Self, rhs: [Int]) -> Bool {
        guard lhs.count == rhs.count else { return false }
        for i in 0..<lhs.count {
            if lhs[i] != rhs[i] { return false }
        }
        return true
    }

    @inlinable
    public static func == (lhs: [Int], rhs: Self) -> Bool {
        guard lhs.count == rhs.count else { return false }
        for i in 0..<lhs.count {
            if lhs[i] != rhs[i] { return false }
        }
        return true
    }

    //--------------------------------------------------------------------------
    // indexing
    @inlinable
    public subscript(index: Int) -> Int {
        get {
            withUnsafeBytes(of: storage) {
                $0.bindMemory(to: Int.self)[index]
            }
        }
        set {
            withUnsafeMutableBytes(of: &storage) {
                $0.bindMemory(to: Int.self)[index] = newValue
            }
        }
    }

    //--------------------------------------------------------------------------
    // Codable
    enum CodingKeys: String, CodingKey { case data }
    
    /// encodes the contents of the array
    public func encode(to encoder: Encoder) throws {
        var container = encoder.unkeyedContainer()
        try forEach {
            try container.encode($0)
        }
    }
    
    // TODO: do a perf test to see if the ManagedBuffer class is faster
    // than using ContiguousArray
    public init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        let rank = MemoryLayout<Storage>.size / MemoryLayout<Element>.size
        var array = ContiguousArray<Int>(repeating: 0, count: rank)
        for i in 0..<rank {
            array[i] = try container.decode(Int.self)
        }
        self.init(array.withUnsafeBytes {
            $0.bindMemory(to: Storage.self)[0]
        })
    }
}

//==============================================================================
//
public extension ShapeArrayProtocol {
    @inlinable
    func map(_ transform: (Element) -> Element) -> Self {
        var result = self
        zip(result.indices, self).forEach { result[$0] = transform($1) }
        return result
    }
    
    @inlinable
    func reduce<Result>(
        _ initialResult: Result,
        _ nextPartialResult: (Result, Element) -> Result) -> Result
    {
        var result = initialResult
        forEach { result = nextPartialResult(result, $0) }
        return result
    }
}

//==============================================================================
//
public protocol ShapeProtocol: Codable {
    // types
    associatedtype Array: ShapeArrayProtocol

    // constants
    static var zeros: Array { get }
    static var ones: Array { get }

    //--------------------------------------------------------------------------
    // properties
    var count: Int { get }
    /// The sparse number of elements spanned by the shape
    var spanCount: Int { get }
    /// The extent of the shape in each dimension
    var extents: Array { get }
    /// The distance to the next element for each dimension
    var strides: Array { get }
        
    //--------------------------------------------------------------------------
    /// Fully specified initializer
    /// - Parameter extents: extent of the shape in each dimension
    /// - Parameter strides: the distance to the next element in each dimension
    init(extents: Array, strides: Array?)
    /// Flattening initializer
    /// - Parameter flattening: the higher order shape to flatten
    init<S>(flattening other: S) where S: ShapeProtocol
    /// Squeezing initializer
    /// - Parameter flattening: the higher order shape to squeeze
    init<S>(squeezing other: S, alongAxes axes: Set<Int>?) where S: ShapeProtocol
}

//==============================================================================
// default implementation
public extension ShapeProtocol {
    //--------------------------------------------------------------------------
    // tuple support
    typealias Tuple = Self.Array.Storage

    @inlinable
    init(extents: Tuple, strides: Tuple? = nil) {
        self.init(extents: Array(extents), strides: Array(strides))
    }
    
    //--------------------------------------------------------------------------
    // computed properties
    /// `true` if the underlying data for the whole shape has a stride of 1.
    @inlinable
    var isContiguous: Bool { count == spanCount }
    /// `true` if the shape has zero elements
    @inlinable
    var isEmpty: Bool { count == 0 }
    /// `true` if the shape has one element
    @inlinable
    var isScalar: Bool { count == 1 }
    /// the index of the last dimension
    @inlinable
    var lastDimension: Int { extents.count - 1 }
    /// the number of sahpe extents
    @inlinable
    var rank: Int { extents.count }
    /// the number of items in extent 0
    @inlinable
    var items: Int { extents[0] }
    /// returns a dense version of self
    @inlinable
    var dense: Self { isContiguous ? self : Self(extents: extents) }

    //--------------------------------------------------------------------------
    // computeSpanCount
    // A sub view may cover a wider range of parent element indexes
    // than the number of dense elements defined by the extents of the view
    // due to striding.
    // The span of the extent is the linear index of the last index + 1
    @inlinable
    static func computeSpanCount(_ extents: Array, _ strides: Array) -> Int {
        (zip(extents, strides).reduce(0) { $0 + ($1.0 - 1) * $1.1 }) + 1
    }
    
    //--------------------------------------------------------------------------
    // init(extents:
    @inlinable
    init(extents: Array) { self.init(extents: extents, strides: nil) }

    //--------------------------------------------------------------------------
    // init(squeezing:
    @inlinable
    init<S>(squeezing other: S, alongAxes axes: Set<Int>? = nil)
        where S: ShapeProtocol
    {
        // make sure we have a positive set of axes to squeeze along
        let rank = Self.zeros.count
        var newExtents = Self.zeros
        var newStrides = Self.zeros
        let axesSet = axes == nil ?
            Set(0..<other.rank) :
            Set(axes!.map { $0 < 0 ? other.rank + $0 : $0 })

        var axis = 0
        for otherAxis in 0..<other.rank where
            !(other.extents[otherAxis] == 1 && axesSet.contains(otherAxis))
        {
            assert(axis < rank,
                   "Unsqueezed axes of `other` exceeds rank of this shape")
            newExtents[axis] = other.extents[otherAxis]
            newStrides[axis] = other.strides[otherAxis]
            axis += 1
        }
        self.init(extents: newExtents, strides: newStrides)
    }
    
    //--------------------------------------------------------------------------
    // equal
    @inlinable
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.extents == rhs.extents
    }

    //--------------------------------------------------------------------------
    // denseStrides
    // computes the strides for a dense shape
    @inlinable
    static func denseStrides(_ extents: Array) -> Array {
        var strides = ones
        for i in stride(from: extents.count - 1, through: 1, by: -1) {
            strides[i - 1] = extents[i] * strides[i]
        }
        return strides
    }
        
    //--------------------------------------------------------------------------
    /// joined
    /// - Parameter others: array of data shapes to join
    /// - Parameter axis: the joining axis
    /// - Returns: returns a new shape that is the join with the others
    @inlinable
    func joined(with others: [Self], alongAxis axis: Int) -> Self {
        var newExtents = extents
        newExtents[axis] += others.reduce(0) { $0 + $1.extents[axis] }
        return Self(extents: newExtents)
    }
    
    //--------------------------------------------------------------------------
    /// makePositive(dims:
    /// The user can specify indices from `-rank..<rank`.
    /// Negative numbers reference dimensions from the end of `extents`
    /// This ensures they are resolved to positive values.
    @inlinable
    static func makePositive(dims: Array) -> Array {
        var positive = dims
        for i in 0..<dims.count where positive[i] < 0 {
            positive[i] += dims.count
        }
        return positive
    }

    //--------------------------------------------------------------------------
    /// linearIndex
    ///    returns the linear element index
    @inlinable
    func linearIndex(of index: Array) -> Int {
        let i = zip(index, strides).reduce(0) { $0 + $1.0 * $1.1 }
        assert(i < spanCount)
        return i
    }

    //--------------------------------------------------------------------------
    /// contains
    @inlinable
    func contains(index: Array) -> Bool {
        linearIndex(of: index) <= spanCount
    }
    
    @inlinable
    func contains(other: Self) -> Bool {
        other.spanCount <= spanCount
    }
    
    @inlinable
    func contains(index: Array, extents: Array) -> Bool {
        linearIndex(of: index) +
            Self(extents: extents, strides: strides).spanCount <= spanCount
    }

    //--------------------------------------------------------------------------
    /// columnMajor
    @inlinable
    var columnMajor: Self {
        // return self if already column major
        guard strides[rank-1] < strides[rank-2] else { return self }
        // compute column major strides for the last 2 dimensions
        var cmExtent = extents
        cmExtent.swapAt(rank-1, rank-2)
        var cmStrides = Self.denseStrides(cmExtent)
        cmStrides.swapAt(rank-1, rank-2)
        return Self(extents: extents, strides: cmStrides)
    }
    
    //--------------------------------------------------------------------------
    /// repeated(to repeatedExtents:
    @inlinable
    func repeated(to repeatedExtents: Array) -> Self {
        // make sure the extents are compatible
        assert({
            for i in 0..<rank {
                if extents[i] != 1 && extents[i] != repeatedExtents[i] {
                    return false
                }
            }
            return true
        }(), "repeated tensor extents must be either 1" +
            " or match the repeated tensor extents")

        // compute strides, setting stride to 0 for repeated dimensions
        var repeatedStrides = Self.zeros
        for i in 0..<rank where repeatedExtents[i] == extents[i] {
            repeatedStrides[i] = strides[i]
        }
        return Self(extents: repeatedExtents, strides: repeatedStrides)
    }

    //--------------------------------------------------------------------------
    /// transposed(with permutations:
    /// Returns a new data shape where the extents and strides are permuted
    /// - Parameter permutations: the indice order mapping. `count` must
    ///   equal `rank`
    /// - Returns: transposed/permuted shape
    /// - Precondition: Each value in `permutations` must be in the range
    ///   `-rank..<rank`
    @inlinable
    func transposed(with permutations: Array? = nil) -> Self {
        guard rank > 1 else { return self }
        var newExtents = extents
        var newStrides = strides

        // determine the new extents and strides
        if let perm = permutations {
            let mapping = Self.makePositive(dims: perm)
            for index in 0..<rank {
                newExtents[index] = extents[mapping[index]]
                newStrides[index] = strides[mapping[index]]
            }
        } else {
            // simple swap
            let r1 = rank-1
            let r2 = rank-2
            newExtents.swapAt(r1, r2)
            newStrides.swapAt(r1, r2)
        }
        return Self(extents: newExtents, strides: newStrides)
    }
}

//==============================================================================
// Shape1
public struct Shape1: ShapeProtocol {
    // constants
    public typealias Array = ShapeArray<(Int)>
    public static let zeros = Array((0))
    public static let ones = Array((1))

    // properties
    public let count: Int
    public let spanCount: Int
    public let extents: Array
    public let strides: Array

    @inlinable
    public init(extents: Array, strides: Array? = nil) {
        self.extents = extents
        self.strides = strides ?? Self.denseStrides(extents)
        count = extents[0]
        spanCount = Self.computeSpanCount(self.extents, self.strides)
    }

    //--------------------------------------------------------------------------
    // init(flattening:
    @inlinable
    public init<S>(flattening other: S) where S: ShapeProtocol {
        self.init(extents: Array((other.count)))
    }
}

//==============================================================================
// Shape2
public struct Shape2: ShapeProtocol {
    // constants
    public typealias Array = ShapeArray<(Int, Int)>
    public static let zeros = Array((0, 0))
    public static let ones = Array((1, 1))

    // properties
    public let count: Int
    public let spanCount: Int
    public let extents: Array
    public let strides: Array

    @inlinable
    public init(extents: Array, strides: Array? = nil) {
        self.extents = extents
        self.strides = strides ?? Self.denseStrides(extents)
        count = extents.reduce(1, *)
        spanCount = Self.computeSpanCount(self.extents, self.strides)
    }

    //--------------------------------------------------------------------------
    // init(flattening:
    @inlinable
    public init<S>(flattening other: S) where S: ShapeProtocol {
        assert(other.isContiguous, "Cannot flatten strided data")
        assert(other.rank >= 2, "you can't flatten from a lower rank")
        self.init(extents: Array((other.extents[0],
                                  other.count / other.extents[0])))
    }
}

//==============================================================================
// Shape3
public struct Shape3: ShapeProtocol {
    // constants
    public typealias Array = ShapeArray<(Int, Int, Int)>
    public static let zeros = Array((0, 0, 0))
    public static let ones = Array((1, 1, 1))

    // properties
    public let count: Int
    public let spanCount: Int
    public let extents: Array
    public let strides: Array

    @inlinable
    public init(extents: Array, strides: Array? = nil) {
        self.extents = extents
        self.strides = strides ?? Self.denseStrides(extents)
        count = extents.reduce(1, *)
        spanCount = Self.computeSpanCount(self.extents, self.strides)
    }
    
    //--------------------------------------------------------------------------
    // init(flattening:
    @inlinable
    public init<S>(flattening other: S) where S: ShapeProtocol {
        assert(other.isContiguous, "Cannot flatten strided data")
        assert(other.rank >= 3, "you can't flatten from a lower rank")
        self.init(extents: Array((
            other.extents[0],
            other.extents[1],
            other.extents[2...].reduce(0,+)
        )))
    }
}

//==============================================================================
// Shape4
public struct Shape4: ShapeProtocol {
    // constants
    public typealias Array = ShapeArray<(Int, Int, Int, Int)>
    public static let zeros = Array((0, 0, 0, 0))
    public static let ones = Array((1, 1, 1, 1))
    
    // properties
    public let count: Int
    public let spanCount: Int
    public let extents: Array
    public let strides: Array
    
    @inlinable
    public init(extents: Array, strides: Array? = nil) {
        self.extents = extents
        self.strides = strides ?? Self.denseStrides(extents)
        count = extents.reduce(1, *)
        spanCount = Self.computeSpanCount(self.extents, self.strides)
    }
    
    //--------------------------------------------------------------------------
    // init(flattening:
    @inlinable
    public init<S>(flattening other: S) where S: ShapeProtocol {
        assert(other.isContiguous, "Cannot flatten strided data")
        assert(other.rank >= 4, "you can't flatten from a lower rank")
        self.init(extents: Array((
            other.extents[0],
            other.extents[1],
            other.extents[2],
            other.extents[3...].reduce(0,+)
        )))
    }
}

//==============================================================================
// Shape5
public struct Shape5: ShapeProtocol {
    // constants
    public typealias Array = ShapeArray<(Int, Int, Int, Int, Int)>
    public static let zeros = Array((0, 0, 0, 0, 0))
    public static let ones = Array((1, 1, 1, 1, 1))
    
    // properties
    public let count: Int
    public let spanCount: Int
    public let extents: Array
    public let strides: Array
    
    @inlinable
    public init(extents: Array, strides: Array? = nil) {
        self.extents = extents
        self.strides = strides ?? Self.denseStrides(extents)
        count = extents.reduce(1, *)
        spanCount = Self.computeSpanCount(self.extents, self.strides)
    }
    
    //--------------------------------------------------------------------------
    // init(flattening:
    @inlinable
    public init<S>(flattening other: S) where S: ShapeProtocol {
        assert(other.isContiguous, "Cannot flatten strided data")
        assert(other.rank >= 5, "you can't flatten from a lower rank")
        self.init(extents: Array((
            other.extents[0],
            other.extents[1],
            other.extents[2],
            other.extents[3],
            other.extents[4...].reduce(0,+)
        )))
    }
}
