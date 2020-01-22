//******************************************************************************
// Copyright 2020 Google LLC
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
/// .. operator
infix operator ..: StridedRangeFormationPrecedence

precedencegroup StridedRangeFormationPrecedence {
    associativity: left
    higherThan: CastingPrecedence
    lowerThan: RangeFormationPrecedence
}

//==============================================================================
/// RangeBound
public protocol RangeBound: Comparable, Numeric {
    func steps(dividedBy step: Self) -> Int
}

public extension RangeBound where Self: FixedWidthInteger {
    @inlinable
    func steps(dividedBy step: Self) -> Int { Int(self / step) }
}

public extension RangeBound where Self: BinaryFloatingPoint {
    @inlinable
    func steps(dividedBy step: Self) -> Int { Int(self / step) }
}

extension Int: RangeBound { }
extension Float: RangeBound { }
extension Double: RangeBound { }

//==============================================================================
/// StridedRangeExpression
public protocol PartialRangeExpression {
    associatedtype Bound: RangeBound

    var step: Bound { get }

    func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
}

extension PartialRangeExpression {
    @inlinable
    public var step: Bound { 1 }
}

//==============================================================================
/// PartialStridedRange
public struct PartialStridedRange<Partial>: PartialRangeExpression
    where Partial: RangeExpression, Partial.Bound: RangeBound
{
    public typealias Bound = Partial.Bound
    public var partialRange: Partial
    public var step: Bound
    
    @inlinable
    public init(partial range: Partial, by step: Bound) {
        self.partialRange = range
        self.step = step
    }
    
    @inlinable
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let r = partialRange.relative(to: collection)
        return StridedRange(from: r.lowerBound, to: r.upperBound, by: step)
    }
}

//==============================================================================
/// range operators
@inlinable
public func .. (range: UnboundedRange, step: Int)
    -> PartialStridedRange<PartialRangeFrom<Int>>
{
    PartialStridedRange(partial: 0..., by: step)
}

// Range with negative bounds
infix operator ..<-: RangeFormationPrecedence

@inlinable
public func ..<- (lower: Int, upper: Int) -> Range<Int> {
    Range(uncheckedBounds: (lower, -upper))
}

// Range through with negative bounds
infix operator ...-: RangeFormationPrecedence

@inlinable
public func ...- (lower: Int, upper: Int) -> ClosedRange<Int> {
    ClosedRange(uncheckedBounds: (lower, -upper))
}

// PartialRangeUpTo/PartialRangeThrough negative
prefix operator ..<-
prefix operator ...-

public extension Int {
    @inlinable
    prefix static func ..<- (upper: Int) -> PartialRangeUpTo<Int> {
        ..<(-upper)
    }
    
    @inlinable
    prefix static func ...- (upper: Int) -> PartialRangeThrough<Int> {
        ...(-upper)
    }
}

// whole range stepped
prefix operator .....

public extension Int {
    @inlinable
    prefix static func ..... (step: Int) ->
        PartialStridedRange<PartialRangeFrom<Int>>
    {
        PartialStridedRange(partial: 0..., by: step)
    }
}

//==============================================================================
/// ..| operator
/// specifies range and relative extent to do windowed operations
infix operator ..|: RangeFormationPrecedence

public extension Int {
    @inlinable
    static func ..| (from: Int, extent: Int) -> Range<Int> {
        Range(uncheckedBounds: (from, from + extent))
    }
}

//==============================================================================
/// StridedRangeExpression
public protocol StridedRangeExpression: PartialRangeExpression { }

//==============================================================================
/// StridedRange
public struct StridedRange<Bound>: StridedRangeExpression, Collection
    where Bound: RangeBound
{
    // properties
    public let count: Int
    public let start: Bound
    public let end: Bound
    public let step: Bound
    
    // open range init
    @inlinable
    public init(from lower: Bound, to upper: Bound, by step: Bound) {
        assert(lower < upper, "Empty range: `to` must be greater than `from`")
        self.count = (upper - lower).steps(dividedBy: step)
        self.start = lower
        self.end = upper
        self.step = step
    }
    
    // closed range init
    @inlinable
    public init(from lower: Bound, through upper: Bound, by step: Bound) {
        assert(lower <= upper,
               "Empty range: `to` must be greater than or equal to `from`")
        let rangeCount = (upper - lower + step)
        self.count = rangeCount.steps(dividedBy: step)
        self.start = lower
        self.end = lower + rangeCount
        self.step = step
    }
    
    @inlinable
    public func relativeTo<C>(_ collection: C) -> Self
        where C : Collection, Bound == C.Index { self }
    
    // Collection
    @inlinable
    public var startIndex: Int { 0 }

    @inlinable
    public var endIndex: Int { count }

    @inlinable
    public subscript(position: Int) -> Bound {
        Bound(exactly: position)! * step
    }

    @inlinable
    public func index(after i: Int) -> Int { i + 1 }
}

//==============================================================================
/// StridedRangeExpression
extension Range: StridedRangeExpression, PartialRangeExpression
    where Bound: RangeBound
{
    @inlinable
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        let start = lowerBound < 0 ? lowerBound + count : lowerBound
        let end = upperBound < 0 ? upperBound + count : upperBound
        return StridedRange(from: start, to: end, by: step)
    }
    
    @inlinable
    public static func .. (r: Self, step: Bound) -> StridedRange<Bound> {
        StridedRange(from: r.lowerBound, to: r.upperBound, by: step)
    }
}

extension ClosedRange: StridedRangeExpression, PartialRangeExpression
    where Bound: RangeBound
{
    @inlinable
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        let start = lowerBound < 0 ? lowerBound + count : lowerBound
        let end = (upperBound < 0 ? upperBound + count : upperBound) + step
        return StridedRange(from: start, to: end, by: step)
    }

    @inlinable
    public static func .. (r: Self, step: Bound) -> StridedRange<Bound> {
        StridedRange(from: r.lowerBound, through: r.upperBound, by: step)
    }
}

extension PartialRangeFrom: PartialRangeExpression where Bound: RangeBound {
    @inlinable
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        let start = lowerBound < 0 ? lowerBound + count : lowerBound
        return StridedRange(from: start, to: count, by: step)
    }

    @inlinable
    public static func .. (range: Self, step: Bound) ->
        PartialStridedRange<Self>
    {
        PartialStridedRange(partial: range, by: step)
    }
}

extension PartialRangeUpTo: PartialRangeExpression where Bound: RangeBound {
    @inlinable
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        let end = upperBound < 0 ? upperBound + count : upperBound
        return StridedRange(from: 0, to: end, by: step)
    }

    @inlinable
    public static func .. (range: Self, step: Bound) ->
        PartialStridedRange<Self>
    {
        PartialStridedRange(partial: range, by: step)
    }
}

extension PartialRangeThrough: PartialRangeExpression
    where Bound: RangeBound
{
    @inlinable
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        let end = (upperBound < 0 ? upperBound + count : upperBound) + step
        return StridedRange(from: 0, to: end, by: step)
    }

    @inlinable
    public static func .. (range: Self, step: Bound) ->
        PartialStridedRange<Self>
    {
        PartialStridedRange(partial: range, by: step)
    }
}

extension Int: PartialRangeExpression {
    public typealias Bound = Int
    
    @inlinable
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        StridedRange(from: self, to: self + 1, by: 1)
    }
}

