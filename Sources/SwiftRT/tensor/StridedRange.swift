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
    func isMultiple(of other: Self) -> Bool
    func steps(dividedBy step: Self) -> Int
}

public extension RangeBound where Self: FixedWidthInteger {
    func steps(dividedBy step: Self) -> Int { Int(self / step) }
}

public extension RangeBound where Self: BinaryFloatingPoint {
    func isMultiple(of other: Self) -> Bool { fmod(self, other) == 0 }
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
    public var step: Bound { 1 }
}

extension RangeExpression where Bound: RangeBound {
    static func .. (range: Self, stride: Bound) -> PartialStridedRange<Self> {
        PartialStridedRange(partial: range, by: stride)
    }
}

//==============================================================================
/// PartialStridedRange
public struct PartialStridedRange<Partial>: PartialRangeExpression
    where Partial: RangeExpression, Partial.Bound: RangeBound
{
    public typealias Bound = Partial.Bound
    public var partialRange: Partial
    public var step: Bound
    
    public init(partial range: Partial, by step: Bound) {
        self.partialRange = range
        self.step = step
    }
    
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let r = partialRange.relative(to: collection)
        return StridedRange(from: r.lowerBound, to: r.upperBound, by: r.step)
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
    public init(from lower: Bound, to upper: Bound, by step: Bound) {
        assert(lower < upper, "Empty range: `to` must be greater than `from`")
        self.count = (upper - lower).steps(dividedBy: step)
        self.start = lower
        self.end = upper
        self.step = step
    }
    
    // closed range init
    public init(from lower: Bound, through upper: Bound, by step: Bound) {
        assert(lower <= upper, "Empty range: `to` must be greater than `from`")
        assert((upper - lower).isMultiple(of: step),
               "Closed ranges must be an even multiple of step")
        self.count = (upper - lower).steps(dividedBy: step)
        self.start = lower
        self.end = (upper + step)
        self.step = step
    }
    
    public func relativeTo<C>(_ collection: C) -> Self
        where C : Collection, Bound == C.Index { self }
    
    // Collection
    public var startIndex: Int { 0 }
    public var endIndex: Int { count }
    public subscript(position: Int) -> Bound {
        Bound(exactly: position)! * step
    }
    public func index(after i: Int) -> Int { i + 1 }
}

//==============================================================================
/// StridedRangeExpression
extension Range: StridedRangeExpression, PartialRangeExpression
    where Bound: RangeBound
{
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        let start = lowerBound < 0 ? lowerBound + count : lowerBound
        let end = upperBound < 0 ? upperBound + count : upperBound
        return StridedRange(from: start, to: end, by: step)
    }
}

extension ClosedRange: StridedRangeExpression, PartialRangeExpression
    where Bound: RangeBound
{
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        let start = lowerBound < 0 ? lowerBound + count : lowerBound
        let end = (upperBound < 0 ? upperBound + count : upperBound) + step
        return StridedRange(from: start, to: end, by: step)
    }
}

extension PartialRangeFrom: PartialRangeExpression where Bound: RangeBound {
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        let start = lowerBound < 0 ? lowerBound + count : lowerBound
        return StridedRange(from: start, to: count, by: step)
    }
}

extension PartialRangeUpTo: PartialRangeExpression where Bound: RangeBound {
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        let end = upperBound < 0 ? upperBound + count : upperBound
        return StridedRange(from: 0, to: end, by: step)
    }
}

extension PartialRangeThrough: PartialRangeExpression
    where Bound: RangeBound
{
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        let end = (upperBound < 0 ? upperBound + count : upperBound) + step
        return StridedRange(from: 0, to: end, by: step)
    }
}

extension Int: PartialRangeExpression {
    public typealias Bound = Int
    
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index {
            StridedRange(from: self, to: self + 1, by: 1)
    }
}

