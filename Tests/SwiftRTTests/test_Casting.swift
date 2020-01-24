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

class test_Casting: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_castElements", test_castElements),
        ("test_flattening", test_flattening),
        ("test_squeezing", test_squeezing),
    ]
    
    //--------------------------------------------------------------------------
    // test_castElements
    func test_castElements() {
        let fMatrix = Matrix(3, 2, with: 0..<6)
        let iMatrix = IndexMatrix(fMatrix)
        XCTAssert(iMatrix == 0..<6)
    }

    //--------------------------------------------------------------------------
    // test_flattening
    func test_flattening() {
        let volume = Volume(2, 3, 4, with: 0..<24)
        
        // volume to matrix
        let matrix = Matrix(flattening: volume)
        XCTAssert(matrix.extents == [2, 12])

        // noop matrix to matrix
        let m2 = Matrix(flattening: matrix)
        XCTAssert(m2.extents == [2, 12])

        // volume to vector
        let v1 = Vector(flattening: volume)
        XCTAssert(v1.extents == [24])

        // matrix to vector
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
}
