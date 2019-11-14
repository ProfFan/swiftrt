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
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    // reduce
    func reduce<T>(x: T,
                   into result: inout T,
                   initialResult: T.Element,
                   along axes: [Int]?,
                   opId: ReductionOp,
                   opNext: @escaping (T.Element, T.Element) -> T.Element,
                   opFinal: @escaping (T.Element) -> T.Element)
        where T: TensorView
    {
        if let axes = axes, axes.count > 0 {
            assert(axes.count <= x.rank, "rank mismatch")
            // TODO
        } else {
            do {
                x.reduce(into: &result, initialResult, opNext)
                let buffer = try result.readWrite()
                buffer[0] = opFinal(buffer[0])
            } catch {
                device.report(error)
            }
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
                   along axes: [Int]?,
                   opId: ReductionOp,
                   opNext: @escaping (T.Element, T.Element) -> T.Element,
                   opFinal: @escaping (T.Element) -> T.Element)
        where T: TensorView
    {
        if let axes = axes, axes.count > 0 {
            assert(axes.count <= x.rank, "rank mismatch")
            queue(#function, { (x.elements(using: self), axes) }, &result)
            { params, result in
                // TODO
            }
        } else {
            queue(#function, { x.elements(using: self) }, &result) {
                $0.reduce(into: &$1, initialResult, opNext)
                $1[$1.startIndex] = opFinal($1[$1.startIndex])
            }
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
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func all<T>(_ x: T, alongAxes axes: [Int]? = nil,
                   result: inout T)
    where T: TensorView, T.Element == Bool
{
    DeviceContext.currentQueue.all(x: x, along: axes, result: &result)
}

/// returns new view
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
public extension TensorView where Element == Bool {
    @inlinable
    func all(alongAxes: Int...) -> Self {
        // turn into a vector
        let axes = shape.makePositive(indices: alongAxes)
        var result = createDense()
        SwiftRT.all(self, alongAxes: axes, result: &result)
        return result
    }
    
    @inlinable
    func all() -> Self {
        var result = createSingleElement()
        SwiftRT.all(self, result: &result)
        return result
    }
    
//    /// returns new view
//    /// - Parameter alongAxes: the axes to operate on
//    /// - Returns: a new NDTensor containing the result
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable
//    func all(squeezing: Int...) -> NDTensor<Element> {
//        let axes = shape.makePositive(indices: squeezing)
//        let axesVec = [Int](with: axes)
//        var result = createDense()
//        SwiftRT.all(self, alongAxes: axesVec, result: &result)
//        return result.squeezed(axes: axes)
//    }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    /// all
    func all<T>(x: T, along axes: [Int]?, result: inout T) where
        T: TensorView, T.Element == Bool
    {
        let xseq = x.elements
        var rseq = result.mutableElements()
        rseq[rseq.startIndex] = xseq.first { $0 == false } == nil
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    /// all
    func all<T>(x: T, along axes: [Int]?, result: inout T) where
        T: TensorView, T.Element == Bool
    {
        queue(#function, { x.elements(using: self) }, &result) {
            $1[$1.startIndex] = $0.first { $0 == false } == nil
        }
    }
}
#endif

//==============================================================================
// >>>>>> User API <<<<<<
/// any(x:alongAxes:)
/// Returns `true` if any value is equal to `true` along the specified
/// axes. Otherwise returns `false`. The result extent along the specified
/// axes will be 1. Rank is not reduced.

/// in place
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func any<T>(_ x: T, alongAxes axes: [Int]? = nil,
                   result: inout T)
    where T: TensorView, T.Element == Bool
{
    DeviceContext.currentQueue.any(x: x, along: axes, result: &result)
}

/// returns new view
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
public extension TensorView where Element == Bool {
    @inlinable
    func any(alongAxes: Int...) -> Self {
        // turn into a vector
        let axes = shape.makePositive(indices: alongAxes)
        var result = createDense()
        SwiftRT.any(self, alongAxes: axes, result: &result)
        return result
    }
    
    @inlinable
    func any() -> Self {
        var result = createSingleElement()
        SwiftRT.any(self, result: &result)
        return result
    }
    
//    /// returns new view
//    /// - Parameter alongAxes: the axes to operate on
//    /// - Returns: a new NDTensor containing the result
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable
//    func any(squeezing: Int...) -> NDTensor<Element> {
//        let axes = shape.makePositive(indices: squeezing)
//        let axesVec = [Int](with: axes)
//        var result = createDense()
//        SwiftRT.any(self, alongAxes: axesVec, result: &result)
//        return result.squeezed(axes: axes)
//    }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    /// any
    func any<T>(x: T, along axes: [Int]?, result: inout T) where
        T: TensorView, T.Element == Bool
    {
        let xseq = x.elements()
        var rseq = result.mutableElements()
        rseq[rseq.startIndex] = xseq.first { $0 == true } != nil
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    /// any
    func any<T>(x: T, along axes: [Int]?, result: inout T) where
        T: TensorView, T.Element == Bool
    {
        queue(#function, { x.elements(using: self) }, &result) {
            $1[$1.startIndex] = $0.first { $0 == true } != nil
        }
    }
}
#endif

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
public func sum<T>(_ x: T, alongAxes axes: [Int]? = nil, result: inout T)
    where T: TensorView, T.Element: Numeric
{
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: T.Element.zero,
                                      along: axes,
                                      opId: .add,
                                      opNext: { $0 + $1 },
                                      opFinal: { $0 })
}

public extension TensorView where Element: AnyNumeric {
    @inlinable
    func sum(alongAxes: Int...) -> Self {
        var result = createDense()
        SwiftRT.sum(self, alongAxes: alongAxes, result: &result)
        return result
    }
    
    @inlinable
    func sum() -> Self {
        var result = createSingleElement()
        SwiftRT.sum(self, result: &result)
        return result
    }
    
//    /// returns new view
//    /// - Parameter alongAxes: the axes to operate on
//    /// - Returns: a new NDTensor containing the result
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable
//    func sum(squeezingAxes: Int...) -> NDTensor<Element> {
//        let axes = shape.makePositive(indices: squeezingAxes)
//        var result = createDense()
//        SwiftRT.sum(self, alongAxes: axes, result: &result)
//        return result.squeezed(axes: axes)
//    }
}

//==============================================================================
// >>>>>> User API <<<<<<
/// mean(x:alongAxes:
/// mean of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func mean<T>(_ x: T, alongAxes axes: [Int]? = nil, result: inout T)
    where T: TensorView, T.Element: FloatingPoint
{
    let count = T.Element(exactly: x.shape.elementCount)!
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: T.Element.zero,
                                      along: axes,
                                      opId: .add,
                                      opNext: { $0 + $1 },
                                      opFinal: { $0 / count })
}

public extension TensorView where Element: FloatingPoint {
    @inlinable
    func mean(alongAxes: Int...) -> Self {
        var result = createDense()
        SwiftRT.mean(self, alongAxes: alongAxes, result: &result)
        return result
    }

    @inlinable
    func mean() -> Self {
        var result = createSingleElement()
        SwiftRT.mean(self, result: &result)
        return result
    }

//    /// returns new view
//    /// - Parameter alongAxes: the axes to operate on
//    /// - Returns: a new NDTensor containing the result
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable
//    func mean(squeezingAxes: Int...) -> NDTensor<Element> {
//        let axes = shape.makePositive(indices: squeezingAxes)
//        var result = createDense()
//        SwiftRT.mean(self, alongAxes: axes, result: &result)
//        return result.squeezed(axes: axes)
//    }
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
public func prod<T>(_ x: T, alongAxes axes: [Int]? = nil, result: inout T)
    where T: TensorView, T.Element: AnyNumeric
{
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: T.Element(any: 1),
                                      along: axes,
                                      opId: .mul,
                                      opNext: { $0 * $1 },
                                      opFinal: { $0 })
}

public extension TensorView where Element: AnyNumeric {
    @inlinable
    func prod(alongAxes: Int...) -> Self {
        var result = createDense()
        SwiftRT.prod(self, alongAxes: alongAxes, result: &result)
        return result
    }
    
    @inlinable
    func prod() -> Self {
        var result = createSingleElement()
        SwiftRT.prod(self, result: &result)
        return result
    }
    
//    /// returns new view
//    /// - Parameter alongAxes: the axes to operate on
//    /// - Returns: a new NDTensor containing the result
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable
//    func prod(squeezingAxes: Int...) -> NDTensor<Element> {
//        let axes = shape.makePositive(indices: squeezingAxes)
//        var result = createDense()
//        SwiftRT.prod(self, alongAxes: axes, result: &result)
//        return result.squeezed(axes: axes)
//    }
}

//==============================================================================
// >>>>>> User API <<<<<<
/// prodNonZeros(x:alongAxes:
/// product of non zero values of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func prodNonZeros<T>(_ x: T, alongAxes axes: [Int]? = nil,
                            result: inout T)
    where T: TensorView, T.Element: AnyNumeric
{
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: T.Element(any: 1),
                                      along: axes,
                                      opId: .mulNonZeros,
                                      opNext: { $1 == 0 ? $0 : $0 * $1 },
                                      opFinal: { $0 })
}

public extension TensorView where Element: AnyNumeric {
    @inlinable
    func prodNonZeros(alongAxes: Int...) -> Self {
        var result = createDense()
        SwiftRT.prodNonZeros(self, alongAxes: alongAxes, result: &result)
        return result
    }
    
    @inlinable
    func prodNonZeros() -> Self {
        var result = createSingleElement()
        SwiftRT.prodNonZeros(self, result: &result)
        return result
    }
    
//    /// returns new view
//    /// - Parameter alongAxes: the axes to operate on
//    /// - Returns: a new NDTensor containing the result
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable
//    func prodNonZeros(squeezingAxes: Int...) -> NDTensor<Element> {
//        let axes = shape.makePositive(indices: squeezingAxes)
//        var result = createDense()
//        SwiftRT.prodNonZeros(self, alongAxes: axes, result: &result)
//        return result.squeezed(axes: axes)
//    }
}

//==============================================================================
// >>>>>> User API <<<<<<
/// minElement(x:alongAxes:
/// returns the minimum element value of `x` along the specified axes
/// TODO: add optional indices
///
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func minElement<T>(_ x: T, alongAxes axes: [Int]? = nil, result: inout T)
    where T: TensorView, T.Element: AnyNumeric & Comparable
{
    let first = try! T.Element(any: x.readOnly()[0])
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: first,
                                      along: axes,
                                      opId: .min,
                                      opNext: { $0 <= $1 ? $0 : $1 },
                                      opFinal: { $0 })
}

public extension TensorView where Element: AnyNumeric  & Comparable {
    @inlinable
    func minElement(alongAxes: Int...) -> Self {
        var result = createDense()
        SwiftRT.minElement(self, alongAxes: alongAxes, result: &result)
        return result
    }
    
    @inlinable
    func minElement() -> Self {
        var result = createSingleElement()
        SwiftRT.minElement(self, result: &result)
        return result
    }
    
//    /// returns new view
//    /// - Parameter alongAxes: the axes to operate on
//    /// - Returns: a new NDTensor containing the result
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable
//    func minElement(squeezingAxes: Int...) -> NDTensor<Element> {
//        let axes = shape.makePositive(indices: squeezingAxes)
//        var result = createDense()
//        SwiftRT.minElement(self, alongAxes: axes, result: &result)
//        return result.squeezed(axes: axes)
//    }
}

//==============================================================================
// >>>>>> User API <<<<<<
/// maxElement(x:alongAxes:
/// returns the maximum element value of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func maxElement<T>(_ x: T, alongAxes axes: [Int]? = nil, result: inout T)
    where T: TensorView, T.Element: AnyNumeric & Comparable
{
    let first = try! T.Element(any: x.readOnly()[0])
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: first,
                                      along: axes,
                                      opId: .max,
                                      opNext: { $0 > $1 ? $0 : $1 },
                                      opFinal: { $0 })
}

public extension TensorView where Element: AnyNumeric  & Comparable {
    @inlinable
    func maxElement(alongAxes: Int...) -> Self {
        var result = createDense()
        SwiftRT.maxElement(self, alongAxes: alongAxes, result: &result)
        return result
    }
    
    @inlinable
    func maxElement() -> Self {
        var result = createSingleElement()
        SwiftRT.maxElement(self, result: &result)
        return result
    }
    
//    /// returns new view
//    /// - Parameter alongAxes: the axes to operate on
//    /// - Returns: a new NDTensor containing the result
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable
//    func maxElement(squeezingAxes: Int...) -> NDTensor<Element> {
//        let axes = shape.makePositive(indices: squeezingAxes)
//        var result = createDense()
//        SwiftRT.maxElement(self, alongAxes: axes, result: &result)
//        return result.squeezed(axes: axes)
//    }
}

//==============================================================================
// >>>>>> User API <<<<<<
/// absmax(x:alongAxes:
/// absolute max of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func absmax<T>(_ x: T, alongAxes axes: [Int]? = nil, result: inout T)
    where T: TensorView, T.Element: AnyNumeric & Comparable
{
    let first = try! T.Element(any: x.readOnly()[0])
    DeviceContext.currentQueue.reduce(
        x: x,
        into: &result,
        initialResult: first,
        along: axes,
        opId: .amax,
        opNext: { $0.magnitude > $1.magnitude ? $0 : $1 },
        opFinal: { $0 })
}

public extension TensorView where Element: AnyNumeric  & Comparable {
    @inlinable
    func absmax(alongAxes: Int...) -> Self {
        var result = createDense()
        SwiftRT.absmax(self, alongAxes: alongAxes, result: &result)
        return result
    }
    
    @inlinable
    func absmax() -> Self {
        var result = createSingleElement()
        SwiftRT.absmax(self, result: &result)
        return result
    }
    
//    /// returns new view
//    /// - Parameter alongAxes: the axes to operate on
//    /// - Returns: a new NDTensor containing the result
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable
//    func absmax(squeezingAxes: Int...) -> NDTensor<Element> {
//        let axes = shape.makePositive(indices: squeezingAxes)
//        var result = createDense()
//        SwiftRT.absmax(self, alongAxes: axes, result: &result)
//        return result.squeezed(axes: axes)
//    }
}

//==============================================================================
// >>>>>> User API <<<<<<
/// abssum(x:alongAxes:
/// Sums the absolute values of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func abssum<T>(_ x: T, alongAxes axes: [Int]? = nil, result: inout T)
    where T: TensorView, T.Element: FloatingPoint
{
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: T.Element.zero,
                                      along: axes,
                                      opId: .asum,
                                      opNext: { $0 + $1.magnitude },
                                      opFinal: { $0 })
}

public extension TensorView where Element: FloatingPoint {
    @inlinable
    func abssum(alongAxes: Int...) -> Self {
        var result = createDense()
        SwiftRT.abssum(self, alongAxes: alongAxes, result: &result)
        return result
    }
    
    @inlinable
    func abssum() -> Self {
        var result = createSingleElement()
        SwiftRT.abssum(self, result: &result)
        return result
    }
    
//    /// returns new view
//    /// - Parameter alongAxes: the axes to operate on
//    /// - Returns: a new NDTensor containing the result
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable
//    func abssum(squeezingAxes: Int...) -> NDTensor<Element> {
//        let axes = shape.makePositive(indices: squeezingAxes)
//        var result = createDense()
//        SwiftRT.abssum(self, alongAxes: axes, result: &result)
//        return result.squeezed(axes: axes)
//    }
}

//==============================================================================
// >>>>>> User API <<<<<<
/// sqrtSumSquares(x:alongAxes:
/// Square root of the sum `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func sqrtSumSquares<T>(_ x: T, alongAxes axes: [Int]? = nil,
                              result: inout T)
    where T: TensorView, T.Element: FloatingPoint
{
    DeviceContext.currentQueue.reduce(x: x,
                                      into: &result,
                                      initialResult: T.Element.zero,
                                      along: axes,
                                      opId: .sqrtSumSquares,
                                      opNext: { $0 + $1 * $1 },
                                      opFinal: { Foundation.sqrt($0) })
}

public extension TensorView where Element: FloatingPoint {
    @inlinable
    func sqrtSumSquares(alongAxes: Int...) -> Self {
        var result = createDense()
        SwiftRT.sqrtSumSquares(self, alongAxes: alongAxes, result: &result)
        return result
    }
    
    @inlinable
    func sqrtSumSquares() -> Self {
        var result = createSingleElement()
        SwiftRT.sqrtSumSquares(self, result: &result)
        return result
    }
    
//    /// returns new view
//    /// - Parameter alongAxes: the axes to operate on
//    /// - Returns: a new NDTensor containing the result
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//    @inlinable
//    func sqrtSumSquares(squeezingAxes: Int...) -> NDTensor<Element> {
//        let axes = shape.makePositive(indices: squeezingAxes)
//        var result = createDense()
//        SwiftRT.sqrtSumSquares(self, alongAxes: axes, result: &result)
//        return result.squeezed(axes: axes)
//    }
}
