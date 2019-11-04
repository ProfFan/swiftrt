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
// >>>>>> User API <<<<<<
/// sum(x:alongAxes:
/// Sums `x` along the specified axes
/// 
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func sum<T>(_ x: T, alongAxes axes: Vector<IndexElement>? = nil,
                   result: inout T)
    where T: TensorView, T.Element: Numeric
{
    DeviceContext.currentQueue.sum(x: x, along: axes, result: &result)
}

public extension TensorView where Element: AnyNumeric {
    @inlinable @inline(__always)
    func sum(alongAxes: Int...) -> Self {
        // turn into a vector
        let axes = Vector<IndexElement>(
            with: shape.makePositive(indices: alongAxes))
        var result = createDense()
        SwiftRT.sum(self, alongAxes: axes, result: &result)
        return result
    }
    
    @inlinable @inline(__always)
    func sum() -> Self {
        let extents = [Int](repeating: 1, count: shape.rank)
        var result = createDense(with: extents)
        SwiftRT.sum(self, result: &result)
        return result
    }
    
    /// returns new view
    /// - Parameter alongAxes: the axes to operate on
    /// - Returns: a new NDTensor containing the result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable @inline(__always)
    func sum(squeezingAxes: Int...) -> NDTensor<Element> {
        let axes = shape.makePositive(indices: squeezingAxes)
        let axesVec = Vector<IndexElement>(with: axes)
        var result = createDense()
        SwiftRT.sum(self, alongAxes: axesVec, result: &result)
        return result.squeezed(axes: axes)
    }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceFunctions {
    // sum
    func sum<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
        T: TensorView, T.Element: Numeric
    {
        if let axes = axes, axes.shape.extents[0] > 0 {
            assert(axes.rank <= x.rank, "rank mismatch")
            // TODO
        } else {
            try! x.values().reduce(to: &result, T.Element.zero) { $0 + $1 }
        }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
// @Target(type:"CPU", appliedTo:"CpuAsynchronousQueue", protocols:[DeviceFunctions])
// target generated from Intent by the compiler
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    // sum
    func sum<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
        T: TensorView, T.Element: Numeric
    {
        if let axes = axes, axes.shape.extents[0] > 0 {
            assert(axes.rank <= x.rank, "rank mismatch")
            queue(#function, { try (x.values(), axes.values()) }, &result)
            { params, result in
                // TODO
            }
        } else {
            queue(#function, { try x.values() }, &result) {
                $0.reduce(to: &$1, T.Element.zero) { $0 + $1 }
            }
        }
    }
}
#endif

