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

class test_Math: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_concat", test_concat),
        ("test_exp", test_exp),
        ("test_log", test_log),
        ("test_neg", test_neg),
        ("test_squared", test_squared),
    ]

    //--------------------------------------------------------------------------
    // test_concat
    func test_concat() {
        let t1 = Matrix(2, 3, with: 1...6)
        let t2 = Matrix(2, 3, with: 7...12)
        let c1 = t1.concat(t2)
        XCTAssert(c1.extents == [4, 3])
        let c1Expected: [Float] = [
            1,  2,  3,
            4,  5,  6,
            7,  8,  9,
            10, 11, 12,
        ]
        XCTAssert(c1.flatArray == c1Expected)
        
        let c2 = t1.concat(t2, alongAxis: 1)
        XCTAssert(c2.extents == [2, 6])
        let c2Expected: [Float] = [
            1, 2, 3,  7,  8,  9,
            4, 5, 6, 10, 11, 12
        ]
        XCTAssert(c2.flatArray == c2Expected)
    }

    //--------------------------------------------------------------------------
    // test_exp
    func test_exp() {
        Platform.local.servicePriority = [cpuSynchronousServiceName]
        let range = 0..<6
        let matrix = Matrix(3, 2, with: range)
        let values = exp(matrix).flatArray
        let expected: [Float] = range.map { Foundation.exp(Float($0)) }
        XCTAssert(values == expected)
        
        let m2 = Matrix(3, 2, with: 1...6)
        XCTAssert(gradientIsValid(at: m2, tolerance: 0.7, in: { exp($0) }))
    }

    //--------------------------------------------------------------------------
    // test_log
    func test_log() {
        Platform.local.servicePriority = [cpuSynchronousServiceName]
        let range = 0..<6
        let matrix = Matrix(3, 2, with: range)
        let values = log(matrix).flatArray
        let expected: [Float] = range.map { Foundation.log(Float($0)) }
        XCTAssert(values == expected)
        
        let m2 = Matrix(3, 2, with: 1...6)
        XCTAssert(gradientIsValid(at: m2, in: { log($0) }))
    }
    
    //--------------------------------------------------------------------------
    // test_neg
    func test_neg() {
        let range = 0..<6
        let matrix = Matrix(3, 2, with: range)
        let expected: [Float] = range.map { -Float($0) }

        let values = matrix.neg().flatArray
        XCTAssert(values == expected)
        
        let values2 = -matrix
        XCTAssert(values2.flatArray == expected)

        let m2 = Matrix(3, 2, with: 1...6)
        XCTAssert(gradientIsValid(at: m2, tolerance: 0.002, in: { neg($0) }))
    }
    
    //--------------------------------------------------------------------------
    // test_squared
    func test_squared() {
        let matrix = Matrix(3, 2, with: [0, -1, 2, -3, 4, 5])
        let values = matrix.squared().flatArray
        let expected: [Float] = (0...5).map { Float($0 * $0) }
        XCTAssert(values == expected)

        let m2 = Matrix(3, 2, with: 1...6)
        XCTAssert(gradientIsValid(at: m2, tolerance: 0.02, in: { squared($0) }))
    }
}
