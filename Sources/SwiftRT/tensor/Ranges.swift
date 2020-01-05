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

//==============================================================================
/// StridedRangeExpression
public protocol StridedRangeExpression: PartialStridedRangeExpression {
    var lowerBound: Bound { get }
    var upperBound: Bound { get }
}

//==============================================================================
/// StridedRange
public struct StridedRange<T: Comparable> {
    public var from: T
    public var to: T
    public var by: T
    public init(from: T, to: T, by: T) {
        assert(from < to, "Empty range: `to` must be greater than `from`")
        self.from = from
        self.to = to
        self.by = by
    }
}

//==============================================================================
/// PartialStridedRangeExpression
public protocol PartialStridedRangeExpression: RangeExpression {
    var stride: Bound { get }
    
    func stridedRangeRelative<C>(to collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
}

public extension PartialStridedRangeExpression where Bound: Numeric {
    var stride: Bound { 1 }
}

public extension PartialStridedRangeExpression {
    /// stridedRangeRelative
    /// this is the default implementation adopted by Swift standard ranges
    func stridedRangeRelative<C>(to collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let range = self.relative(to: collection.indices)
        return StridedRange(from: range.lowerBound,
                            to: range.upperBound,
                            by: stride)
    }
    
    static func .. (range: Self, stride: Bound) -> PartialStridedRange<Self> {
        PartialStridedRange(partial: range, with: stride)
    }
        
    // TODO: do want to support reverse stepping through ranges?
    //    static func ..- (range: Self, stride: Bound) -> PartialStridedRange<Self> {
    //        PartialStridedRange(partial: range, with: -stride)
    //    }
}

//==============================================================================
/// .. stride operator
///
infix operator ..: StridedRangeFormationPrecedence
//infix operator ..-: StridedRangeFormationPrecedence

precedencegroup StridedRangeFormationPrecedence {
    associativity: left
    higherThan: CastingPrecedence
    lowerThan: RangeFormationPrecedence
}

public extension Int {
    static func .. (range: Int, stride: Int) -> PartialStridedRange<Int> {
        assert(stride == 1, "strides are invalid for Integer indexes." +
            " Did you mean to specifiy a closed range with `...`?")
        return PartialStridedRange(partial: range, with: 1)
    }
}

public func .. (range: UnboundedRange, stride: Int)
    -> PartialStridedRange<PartialRangeFrom<Int>>
{
    PartialStridedRange(partial: 0..., with: stride)
}

// TODO: do want to support reverse stepping through ranges?
//public func ..- (range: UnboundedRange, stride: Int)
//    -> PartialStridedRange<PartialRangeFrom<Int>>
//{
//    PartialStridedRange(partial: 0..., with: -stride)
//}

//==============================================================================
/// negative range support
// Range to negative
infix operator ..<-: RangeFormationPrecedence
public func ..<- (lower: Int, upper: Int) -> Range<Int> {
    Range(uncheckedBounds: (lower, -upper))
}

// Range through negative
infix operator ...-: RangeFormationPrecedence
public func ...- (lower: Int, upper: Int) -> ClosedRange<Int> {
    ClosedRange(uncheckedBounds: (lower, -upper))
}

// PartialRangeUpTo/PartialRangeThrough negative
prefix operator ..<-
prefix operator ...-

public extension Int {
    prefix static func ..<- (maximum: Int) -> PartialRangeUpTo<Int> {
        ..<(-maximum)
    }
    
    prefix static func ...- (maximum: Int) -> PartialRangeThrough<Int> {
        ...(-maximum)
    }
}

// whole range stepped
prefix operator .....
//prefix operator .....-

public extension Int {
    prefix static func ..... (stride: Int) ->
        PartialStridedRange<PartialRangeFrom<Int>>
    {
        PartialStridedRange(partial: 0..., with: stride)
    }

// TODO: do want to support reverse stepping through ranges?
//    prefix public static func .....- (stride: Int) ->
//        PartialStridedRange<PartialRangeFrom<Int>>
//    {
//        PartialStridedRange(partial: 0..., with: -stride)
//    }
}

//==============================================================================
/// ..| operator
/// specifies range and relative extent to do windowed operations
infix operator ..|: RangeFormationPrecedence

public extension Int {
    static func ..| (from: Int, extent: Int) -> Range<Int> {
        Range(uncheckedBounds: (from, from + extent))
    }
}

//==============================================================================
/// PartialStridedRangeExpression extensions
extension Range: StridedRangeExpression, PartialStridedRangeExpression
    where Bound: Numeric
{
    public func stridedRangeRelative<C>(to collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        return StridedRange(
            from: lowerBound < 0 ? lowerBound + count : lowerBound,
            to: (upperBound < 0 ? upperBound + count : upperBound),
            by: stride)
    }
}

extension ClosedRange: StridedRangeExpression, PartialStridedRangeExpression
    where Bound: Numeric
{
    public func stridedRangeRelative<C>(to collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        return StridedRange(
            from: lowerBound < 0 ? lowerBound + count : lowerBound,
            to: (upperBound < 0 ? upperBound + count : upperBound) + stride,
            by: stride)
    }
}

extension PartialRangeFrom: PartialStridedRangeExpression where Bound: Numeric {
    public func stridedRangeRelative<C>(to collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        return StridedRange(
            from: lowerBound < 0 ? lowerBound + count : lowerBound,
            to: count,
            by: stride)
    }
}

extension PartialRangeUpTo: PartialStridedRangeExpression where Bound: Numeric {
    public func stridedRangeRelative<C>(to collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        return StridedRange(
            from: 0,
            to: upperBound < 0 ? upperBound + count : upperBound,
            by: stride)
    }
}

extension PartialRangeThrough: PartialStridedRangeExpression
    where Bound: Numeric
{
    public func stridedRangeRelative<C>(to collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        return StridedRange(
            from: 0,
            to: (upperBound < 0 ? upperBound + count : upperBound) + stride,
            by: stride)
    }
}

extension Int: PartialStridedRangeExpression {
    public typealias Bound = Int
    
    public func relative<C>(to collection: C) -> Range<Int>
        where C : Collection, Self.Bound == C.Index {
            Range(uncheckedBounds: (self, self + 1))
    }
    
    public func contains(_ element: Int) -> Bool { element == self }
}

//==============================================================================
/// PartialStridedRange
public struct PartialStridedRange<Partial>: PartialStridedRangeExpression
    where Partial: RangeExpression
{
    public typealias Bound = Partial.Bound
    public var partialRange: Partial
    public var stride: Bound
    
    public init(partial range: Partial, with stride: Bound) {
        self.partialRange = range
        self.stride = stride
    }
    
    public func stridedRangeRelative<C>(to collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let partial = partialRange.relative(to: collection.indices)
        return StridedRange(from: partial.lowerBound,
                            to: partial.upperBound, by: stride)
    }
    
    public func relative<C>(to collection: C) -> Range<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let r = partialRange.relative(to: collection.indices)
        return Range(uncheckedBounds: (r.lowerBound, r.upperBound))
    }
    
    public func contains(_ element: Bound) -> Bool {
        partialRange.contains(element)
    }
}

extension PartialStridedRange: Collection, Sequence
    where Partial: StridedRangeExpression, Partial.Bound: AdditiveArithmetic
{
    public typealias Index = Partial.Bound
    public typealias Element = Partial.Bound

    // Collection
    public var startIndex: Partial.Bound { partialRange.lowerBound }
    public var endIndex: Partial.Bound { partialRange.upperBound }
    public subscript(position: Partial.Bound) -> Partial.Bound { position }
    public func index(after i: Partial.Bound) -> Partial.Bound { i + stride }
}
