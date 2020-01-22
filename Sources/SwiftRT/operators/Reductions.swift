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
import Real

//==============================================================================
// assert messages
public let _messageTensorExtentsMismatch = "tensor extents mismatch"

//==============================================================================
/// NanPropagation
public enum NanPropagation: Int, Codable {
    case propagate, noPropagate
}

//==============================================================================
/// ReductionOp
public enum ReductionOp: Int, Codable {
    case add
    case mean
    case mul
    case min
    case max
    case amax
    case asum
    case sqrtSumSquares
    case mulNonZeros
    case compare
}

public typealias ReduceOpFinal<T: TensorView> = (T.Element) -> T.Element

//==============================================================================
/// all(x:alongAxes:)
/// Returns `true` if all values are equal to `true` along the specified
/// axes. Otherwise returns `false`. The result extent along the specified
/// axes will be 1. Rank is not reduced.

/// in place
/// - Parameter x: value tensor
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func all<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element == Bool
{
    let extents = x.reductionExtents(alongAxes: axes)
    var result = x.createDense(with: extents)
    copy(from: x.view(at: T.Shape.zeros, extents: extents), to: &result)

    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      opId: .compare,
                                      opNext: { $0 && $1 },
                                      opFinal: nil)
    return result
}

/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
public extension TensorView where Element == Bool {
    @inlinable
    func all(alongAxes axes: Set<Int>? = nil) -> Self {
        SwiftRT.all(self, alongAxes: axes)
    }
    
    @inlinable
    func all(alongAxes axes: Int...) -> Self { all(alongAxes: Set(axes)) }
}

//==============================================================================
/// any(x:alongAxes:)
/// Returns `true` if any value is equal to `true` along the specified
/// axes. Otherwise returns `false`. The result extent along the specified
/// axes will be 1. Rank is not reduced.

/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
@inlinable
public func any<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element == Bool
{
    let extents = x.reductionExtents(alongAxes: axes)
    var result = x.createDense(with: extents)
    copy(from: x.view(at: T.Shape.zeros, extents: extents), to: &result)

    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      opId: .compare,
                                      opNext: { $0 || $1 },
                                      opFinal: nil)
    return result
}

/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
public extension TensorView where Element == Bool {
    @inlinable
    func any(alongAxes axes: Set<Int>? = nil) -> Self {
        SwiftRT.any(self, alongAxes: axes)
    }

    @inlinable
    func any(alongAxes axes: Int...) -> Self { any(alongAxes: Set(axes)) }
}

//==============================================================================
/// sum(x:alongAxes:
/// Sums `x` along the specified axes
/// 
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
@inlinable
public func sum<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: Numeric
{
    let extents = x.reductionExtents(alongAxes: axes)
    var result = x.createDense(with: extents).filled(with: T.Element.zero)
    
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      opId: .add,
                                      opNext: +,
                                      opFinal: nil)
    return result
}

public extension TensorView where Element: Numeric {
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func sum(alongAxes axes: Set<Int>? = nil) -> Self {
        SwiftRT.sum(self, alongAxes: axes)
    }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func sum(alongAxes axes: Int...) -> Self { sum(alongAxes: Set(axes)) }
}

//--------------------------------------
// derivative functions
@derivative(of: sum)
@inlinable
internal func _vjpSum<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
    -> (value: T, pullback: (T) -> T) where T: DifferentiableTensorView
{
    let value = x.sum(alongAxes: axes)
    return (value, { [xext = x.extents] in $0.repeated(to: xext) })
}

//==============================================================================
/// mean(x:alongAxes:
/// mean of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
@inlinable
public func mean<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: AlgebraicField
{
    // the divisor is the product of the `axes` that are summed
    let divisor = (axes?.reduce(T.Element.one) {
        $0 * T.Element(exactly: x.extents[$1])!
    }) ?? T.Element(exactly: x.count)!
    
    let extents = x.reductionExtents(alongAxes: axes)
    var result = x.createDense(with: extents).filled(with: T.Element.zero)

    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      opId: .add,
                                      opNext: +,
                                      opFinal: { $0 / divisor })
    return result
}

public extension TensorView where Element: AlgebraicField {
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func mean(alongAxes axes: Set<Int>? = nil) -> Self {
        SwiftRT.mean(self, alongAxes: axes)
    }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func mean(alongAxes axes: Int...) -> Self { mean(alongAxes: Set(axes)) }
}

//--------------------------------------
// derivative functions
@derivative(of: mean)
@inlinable
internal func _vjpMean<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
    -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: AlgebraicField
{
    let value = x.mean(alongAxes: axes)
    let count = T.Element(exactly: x.count)!
    return (value, { [xext = x.extents] in $0.repeated(to: xext) / count })
}

//==============================================================================
/// prod(x:alongAxes:
/// prod of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
@inlinable
public func prod<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: Numeric
{
    let extents = x.reductionExtents(alongAxes: axes)
    var result = x.createDense(with: extents).filled(with: T.Element.one)
    
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      opId: .mul,
                                      opNext: { $0 * $1 },
                                      opFinal: nil)
    return result
}

public extension TensorView where Element: Numeric {
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func prod(alongAxes axes: Set<Int>? = nil) -> Self {
        SwiftRT.prod(self, alongAxes: axes)
    }

    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func prod(alongAxes axes: Int...) -> Self { prod(alongAxes: Set(axes)) }
}

//--------------------------------------
// derivative functions
@derivative(of: prod)
@inlinable
internal func _vjpProd<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
    -> (value: T, pullback: (T) -> T) where T: DifferentiableTensorView
{
    let value = x.prod(alongAxes: axes)
    return (value, { [xext = x.extents] in $0.repeated(to: xext) })
}

//==============================================================================
/// prodNonZeros(x:alongAxes:
/// product of non zero values of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
@inlinable
public func prodNonZeros<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: Numeric
{
    let extents = x.reductionExtents(alongAxes: axes)
    var result = x.createDense(with: extents).filled(with: T.Element.one)

    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      opId: .mulNonZeros,
                                      opNext: { $1 == 0 ? $0 : $0 * $1 },
                                      opFinal: nil)
    return result
}

public extension TensorView where Element: Numeric {
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func prodNonZeros(alongAxes axes: Set<Int>? = nil) -> Self {
        SwiftRT.prodNonZeros(self, alongAxes: axes)
    }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func prodNonZeros(alongAxes axes: Int...) -> Self {
        prodNonZeros(alongAxes: Set(axes))
    }
}

//--------------------------------------
// derivative functions
@derivative(of: prodNonZeros)
@inlinable
internal func _vjpProdNonZeros<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
    -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView
{
    // REVIEW: this is probably wrong
    let value = x.prodNonZeros(alongAxes: axes)
    return (value, { [xext = x.extents] in $0.repeated(to: xext) })
}

//==============================================================================
/// min(x:alongAxes:
/// returns the minimum element value of `x` along the specified axes
/// TODO: add optional indices
///
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
@inlinable
@differentiable(where T: DifferentiableTensorView)
public func min<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: Comparable
{
    let extents = x.reductionExtents(alongAxes: axes)
    var result = x.createDense(with: extents)
    copy(from: x.view(at: T.Shape.zeros, extents: extents), to: &result)

    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      opId: .min,
                                      opNext: { $0 <= $1 ? $0 : $1 },
                                      opFinal: nil)
    return result
}

public extension TensorView where
    Element: Numeric & Comparable & AnyElement
{
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func min(alongAxes axes: Set<Int>? = nil) -> Self {
        SwiftRT.min(self, alongAxes: axes)
    }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func min(alongAxes axes: Int...) -> Self { min(alongAxes: Set(axes)) }
}

//--------------------------------------
// derivative functions
@derivative(of: min)
@inlinable
internal func _vjpMin<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
    -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Comparable
{
    fatalError()
}

//==============================================================================
/// max(x:alongAxes:
/// returns the maximum element value of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
@inlinable
@differentiable(where T: DifferentiableTensorView)
public func max<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: Comparable
{
    let extents = x.reductionExtents(alongAxes: axes)
    var result = x.createDense(with: extents)
    copy(from: x.view(at: T.Shape.zeros, extents: extents), to: &result)

    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      opId: .max,
                                      opNext: { $0 > $1 ? $0 : $1 },
                                      opFinal: nil)
    return result
}

public extension TensorView where
    Element: Numeric & Comparable & AnyElement
{
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func max(alongAxes axes: Set<Int>? = nil) -> Self {
        SwiftRT.max(self, alongAxes: axes)
    }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func max(alongAxes axes: Int...) -> Self { max(alongAxes: Set(axes)) }
}

//--------------------------------------
// derivative functions
@derivative(of: max)
@inlinable
internal func _vjpMax<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
    -> (value: T, pullback: (T) -> T) where
    T: DifferentiableTensorView, T.Element: Comparable
{
    fatalError()
}

//==============================================================================
/// absmax(x:alongAxes:
/// absolute max of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
@inlinable
public func absmax<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: SignedNumeric & Comparable
{
    let extents = x.reductionExtents(alongAxes: axes)
    var result = x.createDense(with: extents)
    copy(from: x.view(at: T.Shape.zeros, extents: extents), to: &result)
    
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      opId: .amax,
                                      opNext: { max(abs($0), abs($1)) },
                                      opFinal: nil)
    return result
}

public extension TensorView where
    Element: SignedNumeric & Comparable & AnyElement
{
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func absmax(alongAxes axes: Set<Int>? = nil) -> Self {
        SwiftRT.absmax(self, alongAxes: axes)
    }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func absmax(alongAxes axes: Int...) -> Self {
        absmax(alongAxes: Set(axes))
    }
}

//--------------------------------------
// derivative functions
@derivative(of: absmax)
@inlinable
internal func _vjpAbsmax<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
    -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: SignedNumeric & Comparable
{
    fatalError()
}

//==============================================================================
/// abssum(x:alongAxes:
/// Sums the absolute values of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
@inlinable
@differentiable(where T: DifferentiableTensorView)
public func abssum<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: SignedNumeric & Comparable
{
    let extents = x.reductionExtents(alongAxes: axes)
    var result = x.createDense(with: extents).filled(with: T.Element.zero)

    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      opId: .asum,
                                      opNext: { $0 + abs($1) },
                                      opFinal: nil)
    return result
}

public extension TensorView where Element: SignedNumeric & Comparable {
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func abssum(alongAxes axes: Set<Int>? = nil) -> Self {
        SwiftRT.abssum(self, alongAxes: axes)
    }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func abssum(alongAxes axes: Int...) -> Self { abssum(alongAxes: Set(axes)) }
}

//--------------------------------------
// derivative functions
@derivative(of: abssum)
@inlinable
internal func _vjpAbsSum<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
    -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: SignedNumeric & Comparable
{
    fatalError()
}

//==============================================================================
/// sqrtSumSquares(x:alongAxes:
/// Square root of the sum `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
@inlinable
public func sqrtSumSquares<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: Real
{
    let extents = x.reductionExtents(alongAxes: axes)
    var result = x.createDense(with: extents).filled(with: T.Element.zero)

    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      opId: .sqrtSumSquares,
                                      opNext: { $0 + $1 * $1 },
                                      opFinal: { .sqrt($0) })
    return result
}

public extension TensorView where Element: Real {
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func sqrtSumSquares(alongAxes axes: Set<Int>? = nil) -> Self {
        SwiftRT.sqrtSumSquares(self, alongAxes: axes)
    }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func sqrtSumSquares(alongAxes axes: Int...) -> Self {
        sqrtSumSquares(alongAxes: Set(axes))
    }
}

//--------------------------------------
// derivative functions
@derivative(of: sqrtSumSquares)
@inlinable
internal func _vjpSqrtSumSquares<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
    -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    fatalError()
}
