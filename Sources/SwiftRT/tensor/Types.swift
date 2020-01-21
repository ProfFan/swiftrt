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
import Numerics

//==============================================================================
/// DifferentiableTensorView
///
/// Marker protocol for `TensorView` that conform to `Differentiable`.
///
/// While this protoocl is not strictly necessary, it is used to reduce the
/// number of generic requirements when writing `@differentiable` attributes on
/// generic differentiable `TensorView` functions.
public protocol DifferentiableTensorView: TensorView & Differentiable where
Self == TangentVector, Element: DifferentiableElement {}

//==============================================================================
/// DifferentiableElement
// this is for shorthand also to make the code less verbose
public protocol DifferentiableElement:
    Differentiable & Numeric where Self == TangentVector {}

extension Float: DifferentiableElement {}
extension Double: DifferentiableElement {}

// this is defined with the typealias because of AD same file
// compiler requirements. Hopefully fixed in the future
extension Complex: DifferentiableElement {
  public typealias TangentVector = Self
}

//==============================================================================
// type extensions
public extension Numeric {
    @inlinable @inline(__always)
    static var one: Self { 1 }
}

extension Complex: AnyElement {
    @inlinable @inline(__always)
    public init() {
        self.init(0)
    }
}

//==============================================================================
public typealias IndexType = Int32
//public typealias Complex = ComplexModule.Complex<Float>

public typealias Vector = VectorType<Float>
public typealias BoolVector = VectorType<Bool>
public typealias IndexVector = VectorType<IndexType>
public typealias ComplexVector = VectorType<Complex<Float>>

public typealias Matrix = MatrixType<Float>
public typealias BoolMatrix = MatrixType<Bool>
public typealias IndexMatrix = MatrixType<IndexType>
public typealias ComplexMatrix = MatrixType<Complex<Float>>

public typealias Volume = VolumeType<Float>
public typealias BoolVolume = VolumeType<Bool>
public typealias IndexVolume = VolumeType<IndexType>
public typealias ComplexVolume = VolumeType<Complex<Float>>
