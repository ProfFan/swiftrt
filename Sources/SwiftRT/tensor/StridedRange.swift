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
/// StridedRangeExpression
public protocol StridedRangeExpression {
    associatedtype Bound: Comparable
    
    // this is intentionally named to avoid conflict with RangeExpression
    func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
}

//==============================================================================
/// StridedRange
public struct StridedRange<Bound>: StridedRangeExpression
    where Bound: Comparable
{
    // properties
    public let start: Bound
    public let end: Bound
    public let step: Bound

    // open range init
    public init(from lower: Bound, to upper: Bound, by step: Bound) {
        assert(lower < upper, "Empty range: `to` must be greater than `from`")
        self.start = lower
        self.end = upper
        self.step = step
    }
    
    public func relativeTo<C>(_ collection: C) -> Self
        where C : Collection, Self.Bound == C.Index { self }
}

extension StridedRange where Bound: AnyNumeric {
    // closed range init
    public init(from lower: Bound, through upper: Bound, by step: Bound) {
        assert(lower <= upper, "Empty range: `to` must be greater than `from`")
        assert(fmod((Float(any: upper) - Float(any: lower)),
                    Float(any: step)) == 0,
               "Closed ranges must be an even multiple of step")
        self.start = lower
        self.end = (upper + step)
        self.step = step
    }
}

//==============================================================================
/// StridedRange Collection extensions
extension StridedRange: Collection, Sequence, IteratorProtocol
    where Bound == Int
{
    public typealias Index = Bound
    public typealias Element = Bound

    // Sequence
    public mutating func next() -> Bound? {
        fatalError()
    }
    
    // Collection
    public var startIndex: Bound { start }
    public var endIndex: Bound { end }
    public subscript(position: Bound) -> Bound { position * step }
    public func index(after i: Bound) -> Bound { i + 1 }
}

extension StridedRange where Bound == Int {
    public var count: Int { (end - start) / step }
}

extension StridedRange where Bound: AnyFloatingPoint {
    public var count: Int { Int(any: (end - start) / step) }
}

//==============================================================================
/// Range extensions
extension Range where Bound: Comparable {
    public static func .. (range: Self, step: Bound) -> StridedRange<Bound> {
        StridedRange(from: range.lowerBound, to: range.upperBound, by: step)
    }
}

extension ClosedRange where Bound: AnyNumeric {
    public static func .. (r: Self, step: Bound) -> StridedRange<Bound> {
        StridedRange(from: r.lowerBound, through: r.upperBound, by: step)
    }
}

//==============================================================================
/// Unbound Range extensions
public func .. (range: UnboundedRange, step: Int)
    -> PartialStridedRange<PartialRangeFrom<Int>>
{
    PartialStridedRange(partial: 0..., by: step)
}

public protocol PartialRangeExpression
    where Self: RangeExpression, Bound: BinaryInteger
{
    static func .. (range: Self, step: Bound) -> PartialStridedRange<Self>
    func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
}

extension PartialRangeExpression
    where Self: RangeExpression, Bound: BinaryInteger
{
    public static func .. (range: Self, step: Bound) -> PartialStridedRange<Self> {
        PartialStridedRange(partial: range, by: step)
    }

    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let r = relative(to: collection.indices)
        return StridedRange(from: r.lowerBound, to: r.upperBound, by: 1)
    }
}

extension PartialRangeFrom: StridedRangeExpression, PartialRangeExpression where Bound: BinaryInteger { }
extension PartialRangeUpTo: StridedRangeExpression, PartialRangeExpression where Bound: BinaryInteger { }
extension PartialRangeThrough: StridedRangeExpression, PartialRangeExpression where Bound: BinaryInteger { }

extension Int: RangeExpression, PartialRangeExpression {
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
    public var step: Bound
    
    public init(partial range: Partial, by step: Bound) {
        self.partialRange = range
        self.step = step
    }

    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let r = partialRange.relative(to: collection.indices)
        return StridedRange(from: r.lowerBound, to: r.upperBound, by: step)
    }
}
