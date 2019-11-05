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
/// neg(x)
/// computes the negated value of `x`
///
/// with placement
/// - Parameter x: value tensor
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func neg<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: FloatingPoint
{
    DeviceContext.currentQueue.neg(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func neg<T>(_ x: T) -> T
    where T: TensorView, T.Element: FloatingPoint
{
    var result = x.createDense()
    neg(x, result: &result)
    return result
}

public extension TensorView where Element: FloatingPoint {
    /// returns new view
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpAbs(_:) where T: TensorFlowFloatingPoint)
    func neg() -> Self {
        var result = createDense()
        SwiftRT.neg(self, result: &result)
        return result
    }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceFunctions {
    /// neg
    /// returns the element-wise negation of `x`
    func neg<T>(x: T, result: inout T) where
        T: TensorView, T.Element: SignedNumeric
    {
        try! x.values().map(to: &result) { -$0 }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
// @Target(type:"CPU", appliedTo:"CpuQueue", protocols:[DeviceFunctions])
// target generated from Intent by the compiler
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    /// neg
    func neg<T>(x: T, result: inout T) where
        T: TensorView, T.Element: SignedNumeric
    {
        queue(#function, { try x.values() }, &result) {
            $0.map(to: &$1) { -$0 }
        }
    }
}
#endif

//==============================================================================
// >>>>>> User API <<<<<<
/// equal
/// Computes `lhs == rhs` element-wise and returns a `TensorView` of Boolean
/// values.
public func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where
    T: TensorView, T.Element: Equatable
{
    assert(lhs.shape == rhs.shape, "shapes must match")
    DeviceContext.currentQueue.equal(lhs: lhs, rhs: rhs, result: &result)
}

public extension TensorView where Element: Equatable {
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand tensor
    /// - Returns: a new tensor containing the result
    @inlinable
    static func .== (_ lhs: Self, _ rhs: Self) -> BoolView {
        var result = lhs.createBoolTensor()
        equal(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
    
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand tensor
    /// - Returns: `true` if the tensors are equal
    @inlinable
    static func == (lhs: Self, rhs: Self) -> Bool {
        // the shapes must match or they are not equal
        guard lhs.shape == rhs.shape else { return false }
        
        // if lhs is an alias for rhs, then they match
        if lhs.tensorArray === rhs.tensorArray &&
            lhs.viewOffset == rhs.viewOffset { return true }
        
        // compare elements
        return try! (lhs .== rhs).all().asElement()
    }

    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand tensor
    /// - Returns: `true` if the tensors are not equal
    @inlinable
    static func != (lhs: Self, rhs: Self) -> Bool {
        return !(lhs == rhs)
    }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceFunctions {
    /// equal
    func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element: Equatable
    {
        try! zip(lhs, rhs).map(to: &result) { $0 == $1 }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
// @Target(type:"CPU", appliedTo:"CpuQueue", protocols:[DeviceFunctions])
// target generated from Intent by the compiler
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    /// equal
    func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element: Equatable
    {
        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
            zip($0.0, $0.1).map(to: &$1) { $0 == $1 }
        }
    }
}
#endif

//==============================================================================
// >>>>>> User API <<<<<<
/// notEqual
/// Computes `lhs != rhs` element-wise and returns a `TensorView` of Boolean
/// values.
public func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where
    T: TensorView, T.Element: Equatable
{
    assert(lhs.shape == rhs.shape, "shapes must match")
    DeviceContext.currentQueue.notEqual(lhs: lhs, rhs: rhs, result: &result)
}

public extension TensorView where Element: Equatable {
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand tensor
    /// - Returns: a new tensor containing the result
    @inlinable
    static func .!= (_ lhs: Self, _ rhs: Self) -> BoolView {
        var result = lhs.createBoolTensor()
        notEqual(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceFunctions {
    /// notEqual
    func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element: Equatable
    {
        try! zip(lhs, rhs).map(to: &result) { $0 != $1 }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
// @Target(type:"CPU", appliedTo:"CpuQueue", protocols:[DeviceFunctions])
// target generated from Intent by the compiler
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    /// equal
    func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element: Equatable
    {
        queue(#function, { try (lhs.values(), rhs.values()) }, &result) {
            zip($0.0, $0.1).map(to: &$1) { $0 != $1 }
        }
    }
}
#endif

