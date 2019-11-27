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
// Adapted from: https://github.com/google/jax/blob/ee36818a589d9c5d82b7b2f636c2f6c88cf9c796/jax/test_util.py#L208
/// verifyGradients
/// - Parameter at:
public func verifyGradients<T>(
    at input: T,
    in body: @differentiable (T) -> T,
    order: Int = 1,
    atol: T = 1e-4,
    rtol: T = 1e-4,
    epsilon: T = 1e-4,
    _ file: String = #file,
    _ line: UInt = #line)
    where
    T: Differentiable & BinaryFloatingPoint,
    T == T.TangentVector,
    T.RawSignificand: FixedWidthInteger
{
    let grad = gradient(at: input, in: body)
    let expectedGrad = numericalJVP(at: input, in: body, epsilon: epsilon)
    // TODO: Replace `expectEqual` and `expectNearlyEqual` calls with something
    // from XCTest.
    print(grad, expectedGrad)
    // expectEqual(value, expectedValue, file: file, line: line)
    // expectNearlyEqual(grad, expectedGrad, file: file, line: line)
}

//==============================================================================
// Adapted from: https://github.com/google/jax/blob/ee36818a589d9c5d82b7b2f636c2f6c88cf9c796/jax/test_util.py#L176
/// numericalJVP
///
public func numericalJVP<T>(
    at input: T,
    in body: @differentiable (T) -> T,
    epsilon: T = 1e-4) -> T
    where
    T : Differentiable & BinaryFloatingPoint,
    T == T.TangentVector,
    T.RawSignificand: FixedWidthInteger
{
    // TODO: Add random `T` argument corresponding to `tangents` and use it to vary `delta`.
    let delta = epsilon
    let valuePos = body(input + delta)
    let valueNeg = body(input - delta)
    return (valuePos - valueNeg) * (0.5 / epsilon)
}
