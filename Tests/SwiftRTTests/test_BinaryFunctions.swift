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
import XCTest
import Foundation

@testable import SwiftRT

class test_BinaryFunctions: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_add", test_add),
        ("test_addScalar", test_addScalar),
        ("test_addAndAssign", test_addAndAssign),

        ("test_subtract", test_subtract),
        ("test_subtractScalar", test_subtractScalar),
        ("test_subtractAndAssign", test_subtractAndAssign),

        ("test_mul", test_mul),
        ("test_mulScalar", test_mulScalar),
        ("test_mulAndAssign", test_mulAndAssign),

        ("test_div", test_div),
        ("test_divScalar", test_divScalar),
        ("test_divAndAssign", test_divAndAssign),
    ]
    
    //--------------------------------------------------------------------------
    // test_add
    func test_add() {
        let m1 = Matrix<Float>((3, 2), name: "matrix", with: 0..<6)
        let m2 = Matrix<Float>((3, 2), name: "matrix", with: 0..<6)
        let result = m1 + m2
        let values = result.array
        let expected: [Float] = [0, 2, 4, 6, 8, 10]
        XCTAssert(values == expected)
    }

    //--------------------------------------------------------------------------
    // test_addScalar
    func test_addScalar() {
        let m1 = Matrix<Float>((3, 2), name: "matrix", with: 1...6)
        let result = m1 + 1
        let values = result.array
        let expected: [Float] = [2, 3, 4, 5, 6, 7]
        XCTAssert(values == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_addAndAssign
    func test_addAndAssign() {
        var m1 = Matrix<Float>((3, 2), name: "matrix", with: 0...5)
        m1 += 2
        let values = m1.array
        let expected: [Float] = [2, 3, 4, 5, 6, 7]
        XCTAssert(values == expected)
    }

    //--------------------------------------------------------------------------
    // test_subtract
    func test_subtract() {
        let m3 = Matrix<Float>((3, 2), name: "matrix", with: 1..<7)
        let m4 = Matrix<Float>((3, 2), name: "matrix", with: 0..<6)
        let result = m3 - m4
        let values = result.array
        let expected: [Float] = [1, 1, 1, 1, 1, 1]
        XCTAssert(values == expected)
    }

    //--------------------------------------------------------------------------
    // test_subtractScalar
    func test_subtractScalar() {
        let m1 = Matrix<Float>((3, 2), name: "matrix", with: 1...6)
        let result = m1 - 1
        let values = result.array
        let expected: [Float] = [0, 1, 2, 3, 4, 5]
        XCTAssert(values == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_subtractAndAssign
    func test_subtractAndAssign() {
        var m1 = Matrix<Float>((3, 2), name: "matrix", with: 1...6)
        m1 -= 1
        let values = m1.array
        let expected: [Float] = [0, 1, 2, 3, 4, 5]
        XCTAssert(values == expected)
    }

    //--------------------------------------------------------------------------
    // test_mul
    func test_mul() {
        let m1 = Matrix<Float>((3, 2), name: "matrix", with: 0..<6)
        let m2 = Matrix<Float>((3, 2), name: "matrix", with: 0..<6)
        let result = m1 * m2
        let values = result.array
        let expected: [Float] = [0, 1, 4, 9, 16, 25]
        XCTAssert(values == expected)
    }

    //--------------------------------------------------------------------------
    // test_mulScalar
    func test_mulScalar() {
        let m1 = Matrix<Float>((3, 2), name: "matrix", with: 1...6)
        let result = m1 * 2
        let values = result.array
        let expected: [Float] = [2, 4, 6, 8, 10, 12]
        XCTAssert(values == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_mulAndAssign
    func test_mulAndAssign() {
        var m1 = Matrix<Float>((3, 2), name: "matrix", with: 1...6)
        m1 *= 2
        let values = m1.array
        let expected: [Float] = [2, 4, 6, 8, 10, 12]
        XCTAssert(values == expected)
    }

    //--------------------------------------------------------------------------
    // test_div
    func test_div() {
        let m1 = Matrix<Float>((3, 2), name: "matrix",
                               with: [1, 4, 9, 16, 25, 36])
        let m2 = Matrix<Float>((3, 2), name: "matrix", with: 1...6)
        let result = m1 / m2
        let values = result.array
        let expected: [Float] = [1, 2, 3, 4, 5, 6]
        XCTAssert(values == expected)
    }

    //--------------------------------------------------------------------------
    // test_divScalar
    func test_divScalar() {
        let m1 = Matrix<Float>((3, 2), name: "matrix", with: 1...6)
        let result = m1 / 2
        let values = result.array
        let expected: [Float] = [0.5, 1, 1.5, 2, 2.5, 3]
        XCTAssert(values == expected)
    }

    //--------------------------------------------------------------------------
    // test_divAndAssign
    func test_divAndAssign() {
        var m1 = Matrix<Float>((3, 2), name: "matrix", with: 1...6)
        m1 /= 2
        let values = m1.array
        let expected: [Float] = [0.5, 1, 1.5, 2, 2.5, 3]
        XCTAssert(values == expected)
    }
}