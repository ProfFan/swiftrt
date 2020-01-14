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

import Complex

//==============================================================================
/// DifferentiableElement
public protocol DifferentiableElement:
    Differentiable & AnyFloatingPoint where Self == TangentVector {}

extension Float: DifferentiableElement {}
extension Double: DifferentiableElement {}

// Note: `DifferentiableElement` is problematic because `Complex` doesn't
// conform to `FloatingPoint` or `AnyFloatingPoint`.
extension Complex: Differentiable {
  public typealias TangentVector = Self
}

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
// TODO: review and discuss how default types are defined
public typealias IndexT = Int32

public typealias Vector = VectorT<Float>
public typealias BoolVector = VectorT<Bool>
public typealias IndexVector = VectorT<IndexT>
public typealias ComplexVector = VectorT<Complex<Float>>

public typealias Matrix = MatrixT<Float>
public typealias BoolMatrix = MatrixT<Bool>
public typealias IndexMatrix = MatrixT<IndexT>
public typealias ComplexMatrix = MatrixT<Complex<Float>>

public typealias Volume = VolumeT<Float>
public typealias BoolVolume = VolumeT<Bool>
public typealias IndexVolume = VolumeT<IndexT>
public typealias ComplexVolume = VolumeT<Complex<Float>>
