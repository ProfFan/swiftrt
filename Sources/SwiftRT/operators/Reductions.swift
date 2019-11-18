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

public typealias ReduceOpFinal<T: TensorView> = (inout T) -> Void

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    // reduce
    func reduce<T>(x: T,
                   into result: inout T,
                   initialResult: T.Element,
                   opId: ReductionOp,
                   opNext: @escaping (T.Element, T.Element) -> T.Element,
                   opFinal: ReduceOpFinal<T>?)
        where T: TensorView
    {
        // fill with the initial result
        fill(&result, with: initialResult)
        
        do {
            // create a temporary view that is repeated to match the input
            var v = try result.sharedView(using: self).repeated(to: x.extents)
            var resultElements = v.mutableElements()
            
            // do the reduction
            zip(x.elements, resultElements.indices).forEach {
                resultElements[$1] = opNext(resultElements[$1], $0)
            }
            opFinal?(&result)
        } catch {
            device.report(error)
        }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    // reduce
    func reduce<T>(x: T,
                   into result: inout T,
                   initialResult: T.Element,
                   opId: ReductionOp,
                   opNext: @escaping (T.Element, T.Element) -> T.Element,
                   opFinal: ReduceOpFinal<T>?)
        where T: TensorView
    {
        // fill with the initial result
        fill(&result, with: initialResult)
        
        do {
            // create a temporary view that is repeated to match the input
            var res = try result.sharedView(using: self).repeated(to: x.extents)
            
            queue(#function, { x.elements(using: self) }, &res) {
                elements, resultElements in
                zip(elements, resultElements.indices).forEach {
                    resultElements[$1] = opNext(resultElements[$1], $0)
                }
                opFinal?(&res)
            }
        } catch {
            device.report(error)
        }
    }
}
#endif

//==============================================================================
// >>>>>> User API <<<<<<
/// all(x:alongAxes:)
/// Returns `true` if all values are equal to `true` along the specified
/// axes. Otherwise returns `false`. The result extent along the specified
/// axes will be 1. Rank is not reduced.

/// in place
/// - Parameter x: value tensor
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func all<T>(_ x: T, result: inout T) where
    T: TensorView, T.Element == Bool
{
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: x.first,
                                      opId: .compare,
                                      opNext: { $0 && $1 },
                                      opFinal: nil)
}

/// returns new view
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
public extension TensorView where Element == Bool {
    @inlinable
    func all(alongAxes axes: Int...) -> Self {
        var result = createReductionResult(alongAxes: axes)
        SwiftRT.all(self, result: &result)
        return result
    }
    
    @inlinable
    func all() -> Self {
        var result = createReductionResult()
        SwiftRT.all(self, result: &result)
        return result
    }
}

//==============================================================================
// >>>>>> User API <<<<<<
/// any(x:alongAxes:)
/// Returns `true` if any value is equal to `true` along the specified
/// axes. Otherwise returns `false`. The result extent along the specified
/// axes will be 1. Rank is not reduced.

/// in place
/// - Parameter x: value tensor
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func any<T>(_ x: T, result: inout T) where
    T: TensorView, T.Element == Bool
{
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: x.first,
                                      opId: .compare,
                                      opNext: { $0 || $1 },
                                      opFinal: nil)
}

/// returns new view
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
public extension TensorView where Element == Bool {
    @inlinable
    func any(alongAxes axes: Int...) -> Self {
        var result = createReductionResult(alongAxes: axes)
        SwiftRT.all(self, result: &result)
        return result
    }
    
    @inlinable
    func any() -> Self {
        var result = createReductionResult()
        SwiftRT.all(self, result: &result)
        return result
    }
}

//==============================================================================
// >>>>>> User API <<<<<<
/// sum(x:alongAxes:
/// Sums `x` along the specified axes
/// 
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func sum<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: Numeric
{
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: T.Element.zero,
                                      opId: .add,
                                      opNext: +,
                                      opFinal: nil)
}

public extension TensorView where Element: Numeric {
    @inlinable
    func sum(alongAxes axes: Int...) -> Self {
        var result = createReductionResult(alongAxes: axes)
        SwiftRT.sum(self, result: &result)
        return result
    }
    
    @inlinable
    func sum() -> Self {
        var result = createReductionResult()
        SwiftRT.sum(self, result: &result)
        return result
    }
}

//==============================================================================
// >>>>>> User API <<<<<<
/// mean(x:alongAxes:
/// mean of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func mean<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: FloatingPoint
{
    let count = T.Element(exactly: x.shape.elementCount)!
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: T.Element.zero,
                                      opId: .add,
                                      opNext: { $0 + $1 },
                                      opFinal: { $0 = $0 / count })
}

public extension TensorView where Element: FloatingPoint {
    @inlinable
    func mean(alongAxes axes: Int...) -> Self {
        var result = createReductionResult(alongAxes: axes)
        SwiftRT.sum(self, result: &result)
        return result
    }
    
    @inlinable
    func mean() -> Self {
        var result = createReductionResult()
        SwiftRT.sum(self, result: &result)
        return result
    }
}

//==============================================================================
// >>>>>> User API <<<<<<
/// prod(x:alongAxes:
/// prod of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func prod<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: Numeric
{
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: T.Element.one,
                                      opId: .mul,
                                      opNext: { $0 * $1 },
                                      opFinal: nil)
}

public extension TensorView where Element: AnyNumeric {
    @inlinable
    func prod(alongAxes axes: Int...) -> Self {
        var result = createReductionResult(alongAxes: axes)
        SwiftRT.sum(self, result: &result)
        return result
    }
    
    @inlinable
    func prod() -> Self {
        var result = createReductionResult()
        SwiftRT.sum(self, result: &result)
        return result
    }
}

//==============================================================================
// >>>>>> User API <<<<<<
/// prodNonZeros(x:alongAxes:
/// product of non zero values of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func prodNonZeros<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: Numeric
{
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: T.Element.one,
                                      opId: .mulNonZeros,
                                      opNext: { $1 == 0 ? $0 : $0 * $1 },
                                      opFinal: nil)
}

public extension TensorView where Element: Numeric {
    @inlinable
    func prodNonZeros(alongAxes axes: Int...) -> Self {
        var result = createReductionResult(alongAxes: axes)
        SwiftRT.sum(self, result: &result)
        return result
    }
    
    @inlinable
    func prodNonZeros() -> Self {
        var result = createReductionResult()
        SwiftRT.sum(self, result: &result)
        return result
    }
}

//==============================================================================
// >>>>>> User API <<<<<<
/// minElement(x:alongAxes:
/// returns the minimum element value of `x` along the specified axes
/// TODO: add optional indices
///
/// - Parameter x: value tensor
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func minElement<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: Numeric & Comparable & AnyElement
{
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: x.first,
                                      opId: .min,
                                      opNext: { $0 <= $1 ? $0 : $1 },
                                      opFinal: nil)
}

public extension TensorView where
    Element: Numeric & Comparable & AnyElement
{
    @inlinable
    func minElement(alongAxes axes: Int...) -> Self {
        var result = createReductionResult(alongAxes: axes)
        SwiftRT.minElement(self, result: &result)
        return result
    }
    
    @inlinable
    func minElement() -> Self {
        var result = createReductionResult()
        SwiftRT.minElement(self, result: &result)
        return result
    }
}

//==============================================================================
// >>>>>> User API <<<<<<
/// maxElement(x:alongAxes:
/// returns the maximum element value of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func maxElement<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: Numeric & Comparable & AnyElement
{
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: x.first,
                                      opId: .max,
                                      opNext: { $0 > $1 ? $0 : $1 },
                                      opFinal: nil)
}

public extension TensorView where
    Element: Numeric & Comparable & AnyElement
{
    @inlinable
    func maxElement(alongAxes axes: Int...) -> Self {
        var result = createReductionResult(alongAxes: axes)
        SwiftRT.maxElement(self, result: &result)
        return result
    }
    
    @inlinable
    func maxElement() -> Self {
        var result = createReductionResult()
        SwiftRT.maxElement(self, result: &result)
        return result
    }
}

//==============================================================================
// >>>>>> User API <<<<<<
/// absmax(x:alongAxes:
/// absolute max of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func absmax<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: Numeric & Comparable & AnyElement
{
    DeviceContext.currentQueue.reduce(
        x: x,
        into: &result,
        initialResult: x.first,
        opId: .amax,
        opNext: { $0.magnitude > $1.magnitude ? $0 : $1 },
        opFinal: nil)
}

public extension TensorView where Element: Numeric & Comparable & AnyElement {
    @inlinable
    func absmax(alongAxes axes: Int...) -> Self {
        var result = createReductionResult(alongAxes: axes)
        SwiftRT.absmax(self, result: &result)
        return result
    }
    
    @inlinable
    func absmax() -> Self {
        var result = createReductionResult()
        SwiftRT.absmax(self, result: &result)
        return result
    }
}

//==============================================================================
// >>>>>> User API <<<<<<
/// abssum(x:alongAxes:
/// Sums the absolute values of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func abssum<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: FloatingPoint
{
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: T.Element.zero,
                                      opId: .asum,
                                      opNext: { $0 + $1.magnitude },
                                      opFinal: nil)
}

public extension TensorView where Element: FloatingPoint {
    @inlinable
    func abssum(alongAxes axes: Int...) -> Self {
        var result = createReductionResult(alongAxes: axes)
        SwiftRT.abssum(self, result: &result)
        return result
    }
    
    @inlinable
    func abssum() -> Self {
        var result = createReductionResult()
        SwiftRT.abssum(self, result: &result)
        return result
    }
}

//==============================================================================
// >>>>>> User API <<<<<<
/// sqrtSumSquares(x:alongAxes:
/// Square root of the sum `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func sqrtSumSquares<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: Real
{
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: T.Element.zero,
                                      opId: .sqrtSumSquares,
                                      opNext: { $0 + $1 * $1 },
                                      opFinal: { $0 = sqrt($0) })
}

public extension TensorView where Element: Real {
    @inlinable
    func sqrtSumSquares(alongAxes axes: Int...) -> Self {
        var result = createReductionResult(alongAxes: axes)
        SwiftRT.sqrtSumSquares(self, result: &result)
        return result
    }
    
    @inlinable
    func sqrtSumSquares() -> Self {
        var result = createReductionResult()
        SwiftRT.sqrtSumSquares(self, result: &result)
        return result
    }
}