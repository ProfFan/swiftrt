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
/// StridedRangeExpression
public protocol StridedRangeExpression: RangeExpression {
    var stride: Bound { get }
    
    func stridedRangeRelative<C>(to collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
}

public extension StridedRangeExpression where Bound == Int {
    var stride: Bound { 1 }
}

public extension StridedRangeExpression {
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

extension RangeExpression where Bound == Int {
    static func .. (range: Self, stride: Bound) -> PartialStridedRange<Self> {
        PartialStridedRange(partial: range, with: stride)
    }
    
// TODO: do want to support reverse stepping through ranges?
//    static func ..- (range: Self, stride: Bound) -> PartialStridedRange<Self> {
//        PartialStridedRange(partial: range, with: -stride)
//    }
}

extension Int {
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

extension Int {
    prefix public static func ..<- (maximum: Int) -> PartialRangeUpTo<Int> {
        ..<(-maximum)
    }
    
    prefix public static func ...- (maximum: Int) -> PartialRangeThrough<Int> {
        ...(-maximum)
    }
}

// whole range stepped
prefix operator .....
//prefix operator .....-

extension Int {
    prefix public static func ..... (stride: Int) ->
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

extension Int {
    static func ..| (from: Int, extent: Int) -> Range<Int> {
        Range(uncheckedBounds: (from, from + extent))
    }
}

//==============================================================================
/// StridedRangeExpression extensions
extension Range: StridedRangeExpression where Bound == Int {
    public func stridedRangeRelative<C>(to collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        StridedRange(
            from: lowerBound < 0 ? lowerBound + collection.count : lowerBound,
            to: (upperBound < 0 ? upperBound + collection.count : upperBound),
            by: stride)
    }
}

extension ClosedRange: StridedRangeExpression where Bound == Int {
    public func stridedRangeRelative<C>(to collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        StridedRange(
            from: lowerBound < 0 ? lowerBound + collection.count : lowerBound,
            to: (upperBound < 0 ? upperBound + collection.count : upperBound) + 1,
            by: stride)
    }
}

extension PartialRangeFrom: StridedRangeExpression where Bound == Int {
    public func stridedRangeRelative<C>(to collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        StridedRange(
            from: lowerBound < 0 ? lowerBound + collection.count : lowerBound,
            to: collection.count,
            by: stride)
    }
}

extension PartialRangeUpTo: StridedRangeExpression where Bound == Int {
    public func stridedRangeRelative<C>(to collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        StridedRange(
            from: 0,
            to: upperBound < 0 ? upperBound + collection.count : upperBound,
            by: stride)
    }
}

extension PartialRangeThrough: StridedRangeExpression where Bound == Int {
    public func stridedRangeRelative<C>(to collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        StridedRange(
            from: 0,
            to: upperBound < 0 ? upperBound + collection.count : upperBound,
            by: stride)
    }
}

extension Int: StridedRangeExpression {
    public typealias Bound = Int
    
    public func relative<C>(to collection: C) -> Range<Int>
        where C : Collection, Self.Bound == C.Index {
            Range(uncheckedBounds: (self, self + 1))
    }
    
    public func contains(_ element: Int) -> Bool { element == self }
}

//==============================================================================
/// PartialStridedRange
public struct PartialStridedRange<Partial>: StridedRangeExpression
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
