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

//==============================================================================
// TODO: review and discuss how default types are defined

public typealias IndexT = Int32

public typealias Vector = VectorT<Float>
public typealias BoolVector = VectorT<Bool>
public typealias IndexVector = VectorT<IndexT>

public typealias Matrix = MatrixT<Float>
public typealias BoolMatrix = MatrixT<Bool>
public typealias IndexMatrix = MatrixT<IndexT>

public typealias Volume = VolumeT<Float>
public typealias BoolVolume = VolumeT<Bool>
public typealias IndexVolume = VolumeT<IndexT>
