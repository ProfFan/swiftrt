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
import SwiftRT

class test_Initialize: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_flattening", test_flattening),
        ("test_squeezing", test_squeezing),
        ("test_cast", test_cast),
        ("test_concatMatrixRows", test_concatMatrixRows),
        ("test_concatMatrixCols", test_concatMatrixCols),
        ("test_repeatElement", test_repeatElement),
        ("test_repeatRowVector", test_repeatRowVector),
        ("test_repeatColVector", test_repeatColVector),
    ]

    //--------------------------------------------------------------------------
    // test_flattening
    func test_flattening() {
        let volume = Volume(2, 3, 4, with: 0..<24)
        let matrix = Matrix(flattening: volume)
        XCTAssert(matrix.extents == [2, 12])

        let v1 = Vector(flattening: volume)
        XCTAssert(v1.extents == [24])

        let v2 = Vector(flattening: matrix)
        XCTAssert(v2.extents == [24])
    }
    
    //--------------------------------------------------------------------------
    // test_squeezing
    func test_squeezing() {
        let volume = Volume(2, 3, 4, with: 0..<24)

        let sumVolumeCols = volume.sum(alongAxes: 2)
        XCTAssert(sumVolumeCols.extents == [2, 3, 1])
        let m0 = Matrix(squeezing: sumVolumeCols)
        XCTAssert(m0.extents == [2, 3])
        
        let sumVolumeRows = volume.sum(alongAxes: 1)
        XCTAssert(sumVolumeRows.extents == [2, 1, 4])
        let m2 = Matrix(squeezing: sumVolumeRows, alongAxes: 1)
        XCTAssert(m2.extents == [2, 4])
        
        // test negative axes
        let m3 = Matrix(squeezing: sumVolumeRows, alongAxes: -2)
        XCTAssert(m3.extents == [2, 4])
    }
    
    //--------------------------------------------------------------------------
    // test_cast
    func test_cast() {
        let fMatrix = Matrix(3, 2, with: 0..<6)
        let iMatrix = IndexMatrix(fMatrix)
        XCTAssert(iMatrix == 0..<6)
    }

    //--------------------------------------------------------------------------
    // test_repeatElement
    func test_repeatElement() {
        let value: Int32 = 42
        let volume = IndexVolume(element: value).repeated(to: (2, 3, 10))
        let expected = [Int32](repeating: value, count: volume.count)
        XCTAssert(volume == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_repeatRowVector
    func test_repeatRowVector() {
        let matrix = IndexMatrix(1, 5, with: 0...4).repeated(to: (5, 5))
        XCTAssert(matrix == [
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4,
        ])
    }
    
    //--------------------------------------------------------------------------
    // test_repeatColVector
    func test_repeatColVector() {
        let matrix = IndexMatrix(5, 1, with: 0...4).repeated(to: (5, 5))
        XCTAssert(matrix == [
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            2, 2, 2, 2, 2,
            3, 3, 3, 3, 3,
            4, 4, 4, 4, 4,
        ])
    }
    
    //--------------------------------------------------------------------------
    // test_concatMatrixRows
    func test_concatMatrixRows() {
        let t1 = Matrix(2, 3, with: 1...6)
        let t2 = Matrix(2, 3, with: 7...12)
        let c3 = Matrix(concatenating: t1, t2)
        XCTAssert(c3.extents == [4, 3])
        XCTAssert(c3 == [
            1,  2,  3,
            4,  5,  6,
            7,  8,  9,
            10, 11, 12,
        ])
    }

    //--------------------------------------------------------------------------
    // test_concatMatrixCols
    func test_concatMatrixCols() {
        let t1 = Matrix(2, 3, with: 1...6)
        let t2 = Matrix(2, 3, with: 7...12)
        let c3 = Matrix(concatenating: t1, t2, alongAxis: 1)
        XCTAssert(c3.extents == [2, 6])
        XCTAssert(c3 == [
            1,  2,  3, 7,  8,  9,
            4,  5,  6, 10, 11, 12,
        ])
    }
}
