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

class test_Shape: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_perfShape2", test_perfShape2),
    ]

    //--------------------------------------------------------------------------
    // test_perfShape2
    func test_perfShape2() {
        #if !DEBUG
        using(Platform.synchronousCpu) {
            var shape = Shape2(extents: Shape2.zeros)
            let index = ShapeArray((1, 1))
            var i = 0
            self.measure {
                for _ in 0..<100000 {
                    let a = Shape2(extents: (3, 4))
                    let b = a.columnMajor
                    let ds = a == b ? b.dense : a.dense
                    let c = Shape2(extents:
                        Shape2.makePositive(dims: Shape2.Array((1, -1))))
                    let r = Shape2(extents: Shape2.ones).repeated(to: a.extents)
                    let j = a.joined(with: [ds, c, r], alongAxis: 1)
                    let t = j.transposed()
                    shape = t
                    i = shape.linearIndex(of: index)
                }
            }
            XCTAssert(shape.extents == Shape2.Array((13, 3)) && i > 0)
        }
        #endif
    }
}
