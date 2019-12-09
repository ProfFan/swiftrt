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
public protocol ShapeArrayProtocol: StaticArrayProtocol, Equatable, Codable
    where Element: BinaryInteger & Equatable, Index == Int
{
    associatedtype Storage
    
    init(_ data: Storage)
    init?(_ data: Storage?)
}

//==============================================================================
//
public struct ShapeArray<Element, Storage> : ShapeArrayProtocol
    where Element: BinaryInteger & Equatable & Codable
{
    public var storage: Storage

    public init(_ data: Storage) {
        storage = data
    }

    //--------------------------------------------------------------------------
    // Equatable
    public static func == (lhs: Self, rhs: Self) -> Bool {
        withUnsafeBytes(of: lhs.storage) { lhsPtr in
            withUnsafeBytes(of: rhs.storage) { rhsPtr in
                memcmp(lhsPtr.baseAddress!,
                       rhsPtr.baseAddress!,
                       MemoryLayout<Storage>.size) == 0
            }
        }
    }

    //--------------------------------------------------------------------------
    // Codable
    enum CodingKeys: String, CodingKey { case data }
    
    /// encodes the contents of the array
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(self, forKey: .data)
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let data = try container.decode(ContiguousArray<Element>.self,
                                        forKey: .data)
        let s: Storage = data.withUnsafeBytes {
            $0.bindMemory(to: Storage.self)[0]
        }
        self.init(s)
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
}

//==============================================================================
// default implementation
public extension ShapeProtocol {
    typealias Tuple = Self.Array.Storage
    
    //--------------------------------------------------------------------------
    // computed properties
    /// `true` if the underlying data for the whole shape has a stride of 1.
    var isContiguous: Bool { count == spanCount }
    /// `true` if the shape has zero elements
    var isEmpty: Bool { count == 0 }
    /// `true` if the shape has one element
    var isScalar: Bool { count == 1 }
    /// the index of the last dimension
    var lastDimension: Int { extents.count - 1 }
    /// the number of sahpe extents
    var rank: Int { extents.count }
    /// the number of items in extent 0
    var items: Int { Int(extents[0]) }
    /// returns a dense version of self
    var dense: Self { isContiguous ? self : Self(extents: extents) }

    //--------------------------------------------------------------------------
    //
    init(extents: Array) { self.init(extents: extents, strides: nil) }

    //--------------------------------------------------------------------------
    // equal
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.extents == rhs.extents
    }

    //--------------------------------------------------------------------------
    // denseStrides
    // computes the strides for a dense shape
    @inlinable
    static func denseStrides(_ extents: Array) -> Array {
        var strides = ones
        stride(from: extents.count, to: 1, by: -1).forEach {
            strides[$0 - 1] = extents[$0] * strides[$0]
        }
        return strides
    }
        
    //--------------------------------------------------------------------------
    /// joined
    /// - Parameter others: array of data shapes to join
    /// - Parameter axis: the joining axis
    /// - Returns: returns a new shape that is the join with the others
    func joined(with others: [Self], alongAxis axis: Int) -> Self {
        var newExtents = extents
        newExtents[axis] += others.reduce(0) { $0 + $1.extents[axis] }
        return Self(extents: newExtents)
    }
    
    //--------------------------------------------------------------------------
    // spanCount
    // A sub view may cover a wider range of parent element indexes
    // than the number of dense elements defined by the extents of the view
    // due to striding.
    // The span of the extent is the linear index of the last index + 1
    @inlinable
    static func spanCount(_ extents: Array, _ strides: Array) -> Int {
        Int(zip(extents, strides).reduce(0) { $0 + ($1.0 - 1) * $1.1 }) + 1
    }
    
    //--------------------------------------------------------------------------
    /// makePositive(dims:
    /// The user can specify indices from `-rank..<rank`.
    /// Negative numbers reference dimensions from the end of `extents`
    /// This ensures they are resolved to positive values.
    func makePositive(dims: Array) -> Array {
        var positive = dims
        let wrapAround = Array.Element(rank)
        for i in 0..<rank where positive[i] < 0 {
            positive[i] += wrapAround
        }
        return positive
    }

    //--------------------------------------------------------------------------
    /// linearIndex
    ///    returns the linear element index
    func linearIndex(of index: Array) -> Int {
        Int(zip(index, strides).reduce(0) { $0 + $1.0 * $1.1 })
    }

    //--------------------------------------------------------------------------
    /// contains
    func contains(offset: Array) -> Bool {
        linearIndex(of: offset) <= spanCount
    }
    
    func contains(other: Self) -> Bool {
        other.spanCount <= spanCount
    }
    
    func contains(offset: Array, extents: Array) -> Bool {
        linearIndex(of: offset) + Self.spanCount(extents, strides) <= spanCount
    }

    //--------------------------------------------------------------------------
    /// columnMajor
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
    func transposed(with permutations: Array? = nil) -> Self {
        guard rank > 1 else { return self }
        var newExtents = extents
        var newStrides = strides

        // determine the new extents and strides
        if let perm = permutations {
            let mapping = makePositive(dims: perm)
            for index in 0..<rank {
                newExtents[index] = extents[Int(mapping[index])]
                newStrides[index] = strides[Int(mapping[index])]
            }
        } else {
            // simple swap
            newExtents.swapAt(rank-1, rank-2)
            newStrides.swapAt(rank-1, rank-2)
        }
        return Self(extents: newExtents, strides: newStrides)
    }
}

//==============================================================================
// Shape2
public struct Shape2: ShapeProtocol {
    // constants
    public typealias Array = ShapeArray<Int, (Int, Int)>
    public static let zeros = Array((0, 0))
    public static let ones = Array((1, 1))

    // properties
    public let count: Int
    public let spanCount: Int
    public let extents: Array
    public let strides: Array

    public init(extents: Array, strides: Array? = nil) {
        self.extents = extents
        self.strides = strides ?? Self.denseStrides(extents)
        count = Int(extents.reduce(1, *))
        spanCount = Self.spanCount(extents, self.strides)
    }
}
