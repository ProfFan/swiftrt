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
/// TensorRange
public struct TensorRange<T> {
    public var start: T
    public var end: T
    public var stride: T
}

//==============================================================================
/// StridedRangeExpression
public protocol StridedRangeExpression: RangeExpression {
    var stride: Bound { get }
    
    func tensorRangeRelative<C>(to collection: C) -> TensorRange<Bound>
        where C : Collection, Self.Bound == C.Index
}

public extension StridedRangeExpression where Bound == Int {
    var stride: Bound { 1 }
}

public extension StridedRangeExpression {
    /// tensorRangeRelative
    /// this is the default implementation adopted by Swift standard ranges
    func tensorRangeRelative<C>(to collection: C) -> TensorRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let range = self.relative(to: collection.indices)
        return TensorRange(start: range.lowerBound,
                           end: range.upperBound,
                           stride: stride)
    }
}

//==============================================================================
/// .. stride operator
///
infix operator ..: StridedRangeFormationPrecedence
precedencegroup StridedRangeFormationPrecedence {
    associativity: left
    higherThan: CastingPrecedence
    lowerThan: RangeFormationPrecedence
}

extension RangeExpression where Bound == Int {
    static func .. (range: Self, stride: Bound) -> PartialStridedRange<Self> {
        PartialStridedRange(partial: range, with: stride)
    }
}

public func .. (range: UnboundedRange, stride: Int)
    -> PartialStridedRange<PartialRangeFrom<Int>>
{
    PartialStridedRange(partial: 0..., with: stride)
}

//==============================================================================
/// .. stride operator
///
infix operator ..|: RangeFormationPrecedence

extension Int {
    static func ..| (from: Int, extent: Int) -> Range<Int> {
        Range(uncheckedBounds: (from, from + extent))
    }
}

//==============================================================================
/// StridedRangeExpression
extension Range: StridedRangeExpression where Bound == Int { }
extension ClosedRange: StridedRangeExpression where Bound == Int { }
extension PartialRangeFrom: StridedRangeExpression where Bound == Int { }
extension PartialRangeUpTo: StridedRangeExpression where Bound == Int { }
extension PartialRangeThrough: StridedRangeExpression where Bound == Int { }

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
    
    public func tensorRangeRelative<C>(to collection: C) -> TensorRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let partial = partialRange.relative(to: collection.indices)
        return TensorRange(start: partial.lowerBound,
                           end: partial.upperBound, stride: stride)
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
