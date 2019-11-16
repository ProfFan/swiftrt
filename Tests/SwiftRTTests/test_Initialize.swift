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

class test_Initialize: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_cast", test_cast),
        ("test_concatMatrixRows", test_concatMatrixRows),
        ("test_concatMatrixCols", test_concatMatrixCols),
//        ("test_repeatElement", test_repeatElement),
        ("test_repeatRowVector", test_repeatRowVector),
        ("test_repeatColVector", test_repeatColVector),
//        ("test_add", test_add),
//        ("test_add", test_add),
//        ("test_add", test_add),
//        ("test_add", test_add),
//        ("test_add", test_add),
    ]
    
    //--------------------------------------------------------------------------
    // test_cast
    func test_cast() {
        let fMatrix = Matrix<Float>(3, 2, with: 0..<6)
        let iMatrix = Matrix<Int32>(fMatrix)
        let expected = [Int32](0..<6)
        XCTAssert(iMatrix.flatArray == expected)
    }

//    //--------------------------------------------------------------------------
//    // test_repeatElement
//    func test_repeatElement() {
//        let value: Int32 = 42
//        let volume = Volume<Int32>((2, 3, 10), repeating: Volume(value))
//        let expected = [Int32](repeating: value, count: volume.elementCount)
//        let values = volume.flatArray
//        XCTAssert(values == expected)
//    }
    
    //--------------------------------------------------------------------------
    // test_repeatRowVector
    func test_repeatRowVector() {
        let matrix = Matrix<Int32>(repeatingRow: 0...4, count: 5)
        let expected: [Int32] = [
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4,
        ]
        let values = matrix.flatArray
        XCTAssert(values == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_repeatColVector
    func test_repeatColVector() {
        let matrix = Matrix<Int32>(repeatingCol: 0...4, count: 5)
        let expected: [Int32] = [
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            2, 2, 2, 2, 2,
            3, 3, 3, 3, 3,
            4, 4, 4, 4, 4,
        ]
        let values = matrix.flatArray
        XCTAssert(values == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_concatMatrixRows
    func test_concatMatrixRows() {
        let t1 = Matrix<Float>(2, 3, with: 1...6)
        let t2 = Matrix<Float>(2, 3, with: 7...12)
        let c3 = Matrix<Float>(concatenating: t1, t2, along: 0)
        XCTAssert(c3.extents == [4, 3])
        let c3Result = c3.flatArray
        let c1Expected: [Float] = [
            1,  2,  3,
            4,  5,  6,
            7,  8,  9,
            10, 11, 12,
        ]
        XCTAssert(c3Result == c1Expected)
    }

    //--------------------------------------------------------------------------
    // test_concatMatrixCols
    func test_concatMatrixCols() {
        let t1 = Matrix<Float>(2, 3, with: 1...6)
        let t2 = Matrix<Float>(2, 3, with: 7...12)
        let c3 = Matrix<Float>(concatenating: t1, t2, along: 0)
        XCTAssert(c3.extents == [4, 3])
        let c3Result = c3.flatArray
        let c1Expected: [Float] = [
            1,  2,  3,
            4,  5,  6,
            7,  8,  9,
            10, 11, 12,
        ]
        XCTAssert(c3Result == c1Expected)
    }
}
