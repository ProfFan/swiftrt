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

class test_Ranges: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_vectorWriteRange", test_vectorWriteRange),
    ]

    //--------------------------------------------------------------------------
    // test_vectorWriteRange
    func test_vectorWriteRange() {
//        Platform.local.servicePriority = [cpuSynchronousServiceName]
//        Platform.log.level = .diagnostic
//        Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
//        var v1 = Vector(with: 0...6)
//        let sevens = Vector(with: repeatElement(7, count: 3))
//        v1[0..<2] = sevens
//        XCTAssert(v1.flatArray == [0, 1, 7, 7, 7, 5])
//        let m2 = Vector(with: 1...6)
//        XCTAssert(gradientIsValid(at: m2, tolerance: 0.7, in: { exp($0) }))
    }
}
