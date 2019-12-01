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
/// gradient
/// finds gradients of tensor
public func gradient<T, R>(
    at x: T,
    in fn: @differentiable (T) -> R) -> T.TangentVector
    where T: DifferentiableTensorView, R: DifferentiableTensorView
{
    return pullback(at: x, in: fn)(R(repeating: 1, like: x))
}

//==============================================================================
/// gradientIsValid
/// - Parameter at:
public func gradientIsValid<T>(
    at input: T,
    in body: @differentiable (T) -> T,
    epsilon: T.Element = T.Element(any: 1e-4),
    tolerance: T.Element = T.Element(any: 5e-4)) -> Bool
    where T: DifferentiableTensorView
{
    let grad = gradient(at: input, in: body)
    let expected = finiteDifferenceJVP(at: input, in: body, epsilon: epsilon)
    let almostEqual = elementsAlmostEqual(grad, expected, tolerance: tolerance)
        .all().element
    if !almostEqual {
        DeviceContext.current[0].writeLog(
            "gradient values do not match numerical jvp values")
        DeviceContext.current[0].writeLog("gradient: \(grad.array)")
        DeviceContext.current[0].writeLog("expected: \(expected.array)")
        let maxDiff = (grad - expected).max().element
        DeviceContext.current[0].writeLog("maxDiff: \(maxDiff)")
    }
    return almostEqual
}

//==============================================================================
/// finiteDifferenceJVP
public func finiteDifferenceJVP<T>(
    at input: T,
    in body: @differentiable (T) -> T,
    epsilon: T.Element = T.Element(any: 1e-4)) -> T
    where T: DifferentiableTensorView
{
    let valuePlus = body(input + epsilon)
    let valueMinus = body(input - epsilon)
    let scale = T.Element(any: 0.5) / epsilon
    return (valuePlus - valueMinus) * scale
}
