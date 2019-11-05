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
        ("test_equality", test_equality),
        ("test_neg", test_neg),
    ]
    
    //--------------------------------------------------------------------------
    // test_equality
    func test_equality() {
        do {
            // compare by value
            let m1 = Matrix<Float>((3, 2), name: "matrix", with: 0..<6)
            let m2 = Matrix<Float>((3, 2), name: "matrix", with: 0..<6)
            XCTAssert(m1 == m2)
            
            // compare via alias detection
            let m3 = m2
            XCTAssert(m3 == m2)
            
            let m4 = Matrix<Float>((3, 2), name: "matrix", with: 1..<7)
            let ne = try (m4 .!= m3).any().asElement()
            XCTAssert(ne)
            XCTAssert(m4 != m3)
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //--------------------------------------------------------------------------
    // test_neg
    func test_neg() {
        do {
            let range = 0..<6
            let matrix = Matrix<Float>((3, 2), name: "matrix", with: range)
            let values = try matrix.neg().array()
            let expected: [Float] = range.map { -Float($0) }
            XCTAssert(values == expected)
        } catch {
            XCTFail(String(describing: error))
        }
    }
}
