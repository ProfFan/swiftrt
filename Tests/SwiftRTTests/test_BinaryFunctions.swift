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
        ("test_addInt32", test_addInt32),
        ("test_addUInt8", test_addUInt8),
        ("test_addScalar", test_addScalar),
        ("test_addAndAssign", test_addAndAssign),

        ("test_subtract", test_subtract),
        ("test_subtractScalar", test_subtractScalar),
        ("test_subtractVector", test_subtractVector),
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
        let m1 = Matrix(3, 2, with: 0..<6)
        let m2 = Matrix(3, 2, with: 0..<6)
        let result = m1 + m2
        let expected: [Float] = [0, 2, 4, 6, 8, 10]
        XCTAssert(result.flatArray == expected)
        XCTAssert(gradientIsValid(at: m1, m2, tolerance: 0.002, in: { $0 + $1 }))
    }

    //--------------------------------------------------------------------------
    // test_addInt32
    func test_addInt32() {
        let m1 = IndexMatrix(3, 2, with: 0..<6)
        let m2 = IndexMatrix(3, 2, with: 0..<6)
        let result = m1 + m2
        let expected: [Int32] = [0, 2, 4, 6, 8, 10]
        XCTAssert(result.flatArray == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_addUInt8
    func test_addUInt8() {
        let m1 = MatrixT<UInt8>(3, 2, with: 0..<6)
        let m2 = MatrixT<UInt8>(3, 2, with: 0..<6)
        let result = m1 + m2
        let expected: [UInt8] = [0, 2, 4, 6, 8, 10]
        XCTAssert(result.flatArray == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_addScalar
    func test_addScalar() {
        let m1 = Matrix(3, 2, with: 1...6)
        let result = m1 + 1
        let expected: [Float] = [2, 3, 4, 5, 6, 7]
        XCTAssert(result.flatArray == expected)

        let result2 = 1 + m1
        XCTAssert(result2.flatArray == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_addAndAssign
    func test_addAndAssign() {
        var m1 = Matrix(3, 2, with: 0...5)
        m1 += 2
        let expected: [Float] = [2, 3, 4, 5, 6, 7]
        XCTAssert(m1.flatArray == expected)
    }

    //--------------------------------------------------------------------------
    // test_subtract
    func test_subtract() {
        let m1 = Matrix(3, 2, with: 1..<7)
        let m2 = Matrix(3, 2, with: 0..<6)
        let result = m1 - m2
        let expected: [Float] = [1, 1, 1, 1, 1, 1]
        XCTAssert(result.flatArray == expected)
        XCTAssert(gradientIsValid(at: m1, m2, tolerance: 0.002, in: { $0 - $1 }))
    }

    //--------------------------------------------------------------------------
    // test_subtractScalar
    func test_subtractScalar() {
        let m1 = Matrix(3, 2, with: 1...6)
        let result = m1 - 1
        let expected: [Float] = [0, 1, 2, 3, 4, 5]
        XCTAssert(result.flatArray == expected)

        let result2 = 1 - m1
        let expected2: [Float] = [0, -1, -2, -3, -4, -5]
        XCTAssert(result2.flatArray == expected2)
    }
    
    //--------------------------------------------------------------------------
    // test_subtractVector
    func test_subtractVector() {
        let m1 = Matrix(3, 2, with: [
            1, 2,
            3, 4,
            5, 6
        ])
        let col = Matrix(3, 1, with: 0...2).repeated(to: (3, 2))

        let result = m1 - col
        let expected: [Float] = [
            1, 2,
            2, 3,
            3, 4
        ]
        XCTAssert(result.flatArray == expected)
        
        let result2 = col - m1
        let expected2: [Float] = [
            -1, -2,
            -2, -3,
            -3, -4
        ]
        XCTAssert(result2.flatArray == expected2)
    }
    
    //--------------------------------------------------------------------------
    // test_subtractAndAssign
    func test_subtractAndAssign() {
        var m1 = Matrix(3, 2, with: 1...6)
        m1 -= 1
        let expected: [Float] = [0, 1, 2, 3, 4, 5]
        XCTAssert(m1.flatArray == expected)
    }

    //--------------------------------------------------------------------------
    // test_mul
    func test_mul() {
        let m1 = Matrix(3, 2, with: 0..<6)
        let m2 = Matrix(3, 2, with: 0..<6)
        let result = m1 * m2
        let expected: [Float] = [0, 1, 4, 9, 16, 25]
        XCTAssert(result.flatArray == expected)
        XCTAssert(gradientIsValid(at: m1, m2, tolerance: 0.006, in: { $0 * $1 }))
    }

    //--------------------------------------------------------------------------
    // test_mulScalar
    func test_mulScalar() {
        let m1 = Matrix(3, 2, with: 1...6)
        let result = m1 * 2
        let expected: [Float] = [2, 4, 6, 8, 10, 12]
        XCTAssert(result.flatArray == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_mulAndAssign
    func test_mulAndAssign() {
        var m1 = Matrix(3, 2, with: 1...6)
        m1 *= 2
        let expected: [Float] = [2, 4, 6, 8, 10, 12]
        XCTAssert(m1.flatArray == expected)
    }

    //--------------------------------------------------------------------------
    // test_div
    func test_div() {
        let m1 = Matrix(3, 2, with: [1, 4, 9, 16, 25, 36])
        let m2 = Matrix(3, 2, with: 1...6)
        let result = m1 / m2
        let expected: [Float] = [1, 2, 3, 4, 5, 6]
        XCTAssert(result.flatArray == expected)
        XCTAssert(gradientIsValid(at: m1, m2, tolerance: 0.002, in: { $0 / $1 }))
    }

    //--------------------------------------------------------------------------
    // test_divScalar
    func test_divScalar() {
        let m1 = Matrix(3, 2, with: 1...6)
        let result = m1 / 2
        let expected: [Float] = [0.5, 1, 1.5, 2, 2.5, 3]
        XCTAssert(result.flatArray == expected)
    }

    //--------------------------------------------------------------------------
    // test_divAndAssign
    func test_divAndAssign() {
        var m1 = Matrix(3, 2, with: 1...6)
        m1 /= 2
        let expected: [Float] = [0.5, 1, 1.5, 2, 2.5, 3]
        XCTAssert(m1.flatArray == expected)
    }
}
