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

class test_ElementWise: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_concat", test_concat),
        ("test_equality", test_equality),
        ("test_log", test_log),
        ("test_neg", test_neg),
        ("test_maximum", test_maximum),
        ("test_maximumScalar", test_maximumScalar),
        ("test_minimum", test_minimum),
        ("test_minimumScalar", test_minimumScalar),
        ("test_squared", test_squared),
    ]

    //--------------------------------------------------------------------------
    // test_concat
    func test_concat() {
//        Platform.local.servicePriority = [cpuSynchronousServiceName]
        
        let t1 = Matrix<Float>((2, 3), with: 1...6)
        let t2 = Matrix<Float>((2, 3), with: 7...12)
        let c1 = t1.concat(t2)
        XCTAssert(c1.extents == [4, 3])
        let c1Result = c1.array
        let c1Expected: [Float] = [
            1,   2, 3,
            4,   5, 6,
            7,   8, 9,
            10, 11, 12,
        ]
        XCTAssert(c1Result == c1Expected)

        let c2 = t1.concat(t2, along: 1)
        XCTAssert(c2.extents == [2, 6])
        let c2Result = c2.array
        let c2Expected: [Float] = [
            1, 2, 3, 7, 8, 9,
            4, 5, 6, 10, 11, 12
        ]
        XCTAssert(c2Result == c2Expected)
        
        let c3 = Matrix<Float>(concatenating: t1, t2, along: 0)
        XCTAssert(c3.extents == [4, 3])
        let c3Result = c3.array
        XCTAssert(c3Result == c1Expected)
    }

    //--------------------------------------------------------------------------
    // test_equality
    func test_equality() {
        // compare by value
        let m1 = Matrix<Float>((3, 2), name: "matrix", with: 0..<6)
        let m2 = Matrix<Float>((3, 2), name: "matrix", with: 0..<6)
        XCTAssert(m1 == m2)
        
        // compare via alias detection
        let m3 = m2
        XCTAssert(m3 == m2)
        
        let m4 = Matrix<Float>((3, 2), name: "matrix", with: 1..<7)
        let ne = (m4 .!= m3).any().element
        XCTAssert(ne)
        XCTAssert(m4 != m3)
    }

    //--------------------------------------------------------------------------
    // test_log
    func test_log() {
        let range = 0..<6
        let matrix = Matrix<Float>((3, 2), name: "matrix", with: range)
        let values = log(matrix).array
        let expected: [Float] = range.map { Foundation.log(Float($0)) }
        XCTAssert(values == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_neg
    func test_neg() {
        let range = 0..<6
        let matrix = Matrix<Float>((3, 2), name: "matrix", with: range)
        let expected: [Float] = range.map { -Float($0) }

        let values = matrix.neg().array
        XCTAssert(values == expected)
        
        let values2 = -matrix
        XCTAssert(values2.array == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_squared
    func test_squared() {
        let matrix = Matrix<Float>((3, 2), name: "matrix",
                                   with: [0, -1, 2, -3, 4, 5])
        let values = matrix.squared().array
        let expected: [Float] = (0...5).map { Float($0 * $0) }
        XCTAssert(values == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_maximum
    func test_maximum() {
        let m1 = Matrix<Float>((3, 2), name: "matrix",
                                   with: [0, 1, -2, -3, -4, 5])
        let m2 = Matrix<Float>((3, 2), name: "matrix",
                                   with: [0, -7, 2, 3, 4, 5])
        let result = maximum(m1, m2)
        let values = result.array
        let expected: [Float] = [0, 1, 2, 3, 4, 5]
        XCTAssert(values == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_maximumScalar
    func test_maximumScalar() {
        let m1 = Matrix<Float>((3, 2), name: "matrix", with: 0...5)
        let result = maximum(m1, 2)
        let values = result.array
        let expected: [Float] = [2, 2, 2, 3, 4, 5]
        XCTAssert(values == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_minimum
    func test_minimum() {
        let m1 = Matrix<Float>((3, 2), name: "matrix",
                               with: [0, 1, 2, -3, 4, -5])
        let m2 = Matrix<Float>((3, 2), name: "matrix",
                               with: [0, -1, -2, 3, -4, 5])
        let result = minimum(m1, m2)
        let values = result.array
        let expected: [Float] = [0, -1, -2, -3, -4, -5]
        XCTAssert(values == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_minimumScalar
    func test_minimumScalar() {
        let m1 = Matrix<Float>((3, 2), name: "matrix", with: 0...5)
        let result = minimum(m1, 3)
        let values = result.array
        let expected: [Float] = [0, 1, 2, 3, 3, 3]
        XCTAssert(values == expected)
    }
}
