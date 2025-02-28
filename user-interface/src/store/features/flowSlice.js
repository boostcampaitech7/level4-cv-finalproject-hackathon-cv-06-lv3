import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;

//
// -------------------- Flows API Thunks --------------------
//

// 특정 프로젝트의 Flows 가져오기
export const fetchFlowsByProject = createAsyncThunk(
  'flows/fetchFlowsByProject',
  async (projectId, { rejectWithValue }) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/flows/?project_id=${projectId}`
      );
      if (!response.ok) throw new Error('Failed to fetch flows');

      const data = await response.json();

      if (!Array.isArray(data.flows)) {
        throw new Error('Invalid response format: flows is not an array');
      }

      // ✅ "flows" 키 아래의 배열을 변환 (id → flowId)
      const formattedFlows = Object.fromEntries(
        data.flows.map(flow => [
          flow.id,
          { ...flow, flowId: flow.id, projectId },
        ])
      );
      return { projectId, flows: formattedFlows };
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

// Flow 추가
export const addFlowAsync = createAsyncThunk(
  'flows/addFlow',
  async ({ projectId, flowName }, { rejectWithValue }) => {
    try {
      const response = await fetch(`${API_BASE_URL}/flows/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_id: projectId, flow_name: flowName }),
      });

      if (!response.ok) throw new Error('Failed to add flow');

      const newFlow = await response.json();

      return {
        flowId: newFlow.flow_id,
        flow: {
          flowId: newFlow.flow_id,
          projectId,
          flow_name: flowName, // ✅ flow_name 저장
        },
      };
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

// Flow 수정
export const editFlowAsync = createAsyncThunk(
  'flows/editFlow',
  async ({ flowId, flowName }, { rejectWithValue }) => {
    try {
      const response = await fetch(`${API_BASE_URL}/flows/`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flow_id: flowId, flow_name: flowName }),
      });

      if (!response.ok) throw new Error('Failed to edit flow');

      return { flowId, flowName };
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

// Flow 삭제
export const deleteFlowAsync = createAsyncThunk(
  'flows/deleteFlow',
  async ({ flowId }, { rejectWithValue }) => {
    try {
      const response = await fetch(`${API_BASE_URL}/flows/`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flow_id: flowId }),
      });

      if (!response.ok) throw new Error('Failed to delete flow');

      return flowId;
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

//
// -------------------- Dataset 관련 Thunks 추가 --------------------
//

// ✅ 특정 Flow에 추가된 CSV 목록 조회
export const fetchFlowDatasets = createAsyncThunk(
  'flows/fetchFlowDatasets',
  async (flowId, { rejectWithValue }) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/flows/csv-add/?flow_id=${flowId}`
      );
      if (!response.ok) throw new Error('Failed to fetch flow datasets');

      const data = await response.json();

      // 🔍 응답 검증
      if (!data.csvs || !Array.isArray(data.csvs)) {
        console.error(
          '❌ API response does not contain a valid csvs array:',
          data
        );
        return rejectWithValue('Invalid API response');
      }

      // ✅ "csvs" 배열을 변환하여 flow에 저장
      const formattedDatasets = data.csvs.map(({ id, csv_name }) => {
        if (typeof csv_name !== 'string') {
          console.error('❌ Invalid csv_name:', csv_name);
          return { csvId: id, fileName: 'Unknown' }; // 기본값 설정
        }

        return {
          csvId: id,
          fileName: csv_name,
        };
      });

      return { flowId, datasets: formattedDatasets };
    } catch (error) {
      console.error('❌ fetchFlowDatasets Error:', error);
      return rejectWithValue(error.message);
    }
  }
);

// ✅ Flow에 CSV 추가
export const addCsvToFlow = createAsyncThunk(
  'flows/addCsvToFlow',
  async ({ flowId, csvIds }, { rejectWithValue }) => {
    try {
      const response = await fetch(`${API_BASE_URL}/flows/csv-add/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flow_id: parseInt(flowId), csv_ids: csvIds }),
      });

      if (!response.ok) throw new Error('Failed to add CSV to flow');

      return { flowId, csvIds };
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

// ✅ 특정 Flow의 properties 조회
export const fetchFlowProperties = createAsyncThunk(
  'flows/fetchFlowProperties',
  async (flowId, { rejectWithValue }) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/concat-columns/properties/?flow_id=${flowId}`
      );
      if (!response.ok) throw new Error('Failed to fetch properties');

      const data = await response.json();

      return { flowId, data };
    } catch (error) {
      console.error('❌ Error fetching properties:', error);
      return rejectWithValue(error.message);
    }
  }
);

export const fetchPropertyTypes = createAsyncThunk(
  'flows/fetchPropertyTypes',
  async (flowId, { rejectWithValue }) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/concat-columns/types/?flow_id=${flowId}`
      );
      if (!response.ok) throw new Error('Failed to fetch properties');

      const data = await response.json();

      return { flowId, data };
    } catch (error) {
      console.error('❌ Error fetching properties:', error);
      return rejectWithValue(error.message);
    }
  }
);

// 데이터 타입 변경 (숫자형, 문자형, 문자, 사용 불가)
export const savePropertyTypes = createAsyncThunk(
  'flows/savePropertyTypes',
  async ({ flowId, update }, { rejectWithValue }) => {
    try {
      const response = await fetch(`${API_BASE_URL}/concat-columns/types/`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(update),
      });

      if (!response.ok) throw new Error('Failed to update property types');

      return { flowId, update };
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

// ✅ 특정 Flow의 히스토그램 데이터 가져오기
export const fetchFlowHistograms = createAsyncThunk(
  'flows/fetchFlowHistograms',
  async (flowId, { rejectWithValue }) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/histograms/all?flow_id=${flowId}`
      );
      if (!response.ok) throw new Error('Failed to fetch histograms');

      const data = await response.json();

      return { flowId, histograms: data.histograms };
    } catch (error) {
      console.error('❌ Error fetching histograms:', error);
      return rejectWithValue(error.message);
    }
  }
);

export const fetchPropertyHistograms = createAsyncThunk(
  'flows/fetchPropertyHistograms',
  async ({ flowId, column_name }, thunkAPI) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/histograms/?flow_id=${flowId}&column_name=${column_name}`
        //`${API_BASE_URL}/flows/concat-csv-column/?flow_id=${flowId}&column_name=${column_name}`
      );
      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }
      const data = await response.json();
      return { flowId, column_name, histograms: data.histograms };
    } catch (error) {
      return thunkAPI.rejectWithValue(error.message);
    }
  }
);

// ✅ 새로운 카테고리 정보 PUT 요청 (Next Step 버튼 클릭 시 실행)
export const savePropertyCategories = createAsyncThunk(
  'flows/savePropertyCategories',
  async ({ flowId, update }, { rejectWithValue }) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/concat-columns/properties/`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(update),
        }
      );

      if (!response.ok) throw new Error('Failed to update property categories');

      return { flowId, update };
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const fetchOptimizationData = createAsyncThunk(
  'flows/fetchOptimizationData',
  async ({ flowId, property, type }, thunkAPI) => {
    // type에 따라 endpoint 선택
    try {
      const response = await fetch(
        `${API_BASE_URL}/optimization/goals/?flow_id=${flowId}&column_name=${property}`
      );
      if (!response.ok) {
        throw new Error('Failed to get optimization data');
      }
      const data = await response.json();
      // data 예시: { min: 10, max: 100, goal: "No Optimization" } 혹은 goal이 없으면 기본값으로 대체
      return { flowId, property, type, data };
    } catch (error) {
      return thunkAPI.rejectWithValue(error.message);
    }
  }
);

export const postOptimizationData = createAsyncThunk(
  'flows/postOptimizationData',
  async (
    { flowId, property, type, goal, minimum_value, maximum_value },
    thunkAPI
  ) => {
    const goalMapping = {
      'No Optimization': 1,
      Maximize: 2,
      Minimize: 3,
      'Fit to Range': 4,
      'Fit to Property': 5,
    };

    let payload = {
      flow_id: flowId,
      column_name: property,
      optimize_goal: goalMapping[goal],
      minimum_value: minimum_value,
      maximum_value: maximum_value,
    };

    try {
      const response = await fetch(`${API_BASE_URL}/optimization/goals/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(`POST failed: ${response.statusText}`);
      }
      const data = await response.json();
      return { flowId, property, type, payload, data };
    } catch (error) {
      return thunkAPI.rejectWithValue(error.message);
    }
  }
);

// ✅ 우선순위 저장 API 호출
export const postOptimizationOrder = createAsyncThunk(
  'flows/postOptimizationOrder',
  async ({ flowId, priorities }, { rejectWithValue }) => {
    try {
      const responses = await Promise.all(
        priorities.map((column_name, index) =>
          fetch(`${API_BASE_URL}/optimization/orders/`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              flow_id: flowId,
              column_name,
              optimize_order: index + 1, // 1부터 시작하는 순서
            }),
          }).then(async response => {
            if (!response.ok) {
              // 응답이 ok가 아닐 경우 에러 메시지 추출
              const errorData = await response.json();
              throw new Error(errorData.message || 'Fetch error');
            }
            return response.json();
          })
        )
      );
      return responses;
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const createModelThunk = createAsyncThunk(
  'flows/createModel',
  async (flowId, { rejectWithValue }) => {
    try {
      const response = await fetch(`${API_BASE_URL}/processing/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ flow_id: flowId }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.message || `HTTP error! status: ${response.status}`
        );
      }
      const data = await response.json();
      return data;
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const pollFlowProgress = (flowId, toast) => dispatch => {
  const intervalId = setInterval(async () => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/flows/progress/?flow_id=${flowId}`
      );
      const data = await response.json();
      dispatch(updateFlow({ flowId: data.flow_id, progress: data.progress }));
    } catch (error) {
      console.error('Failed to fetch progress:', error);
      toast({
        title: 'Error fetching progress',
        description: error.message,
        status: 'error',
      });
    }
  }, 3000);
  return intervalId;
};

// 1. Feature Importance를 가져오는 thunk
export const fetchSurrogateFeatureImportance = createAsyncThunk(
  'flows/fetchSurrogateFeatureImportance',
  async (flowId, thunkAPI) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/surrogate/feature-importance/?flow_id=${flowId}`
      );
      if (!response.ok) {
        throw new Error('Failed to fetch surrogate feature importance');
      }
      const data = await response.json();
      // data.surrogate_feature_importance는 배열 형태로 반환됨.
      return { flowId, data: data.feature_importance };
    } catch (error) {
      return thunkAPI.rejectWithValue(error.message);
    }
  }
);

// 2. Matric(혹은 Metrics)를 가져오는 thunk
export const fetchSurrogateMatric = createAsyncThunk(
  'flows/fetchSurrogateMatric',
  async (flowId, thunkAPI) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/surrogate/matric/?flow_id=${flowId}`
      );
      if (!response.ok) {
        throw new Error('Failed to fetch surrogate matric');
      }
      const data = await response.json();

      // data.surrogate_matric는 배열 형태
      return { flowId, data: data.surrogate_matric };
    } catch (error) {
      return thunkAPI.rejectWithValue(error.message);
    }
  }
);

// 3. Surrogate Result를 가져오는 thunk
export const fetchSurrogateResult = createAsyncThunk(
  'flows/fetchSurrogateResult',
  async (flowId, thunkAPI) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/surrogate/result/?flow_id=${flowId}`
      );
      if (!response.ok) {
        throw new Error('Failed to fetch surrogate result');
      }
      const data = await response.json();
      // data.surrogate_result는 배열 형태
      return { flowId, data: data.surrogate_result };
    } catch (error) {
      return thunkAPI.rejectWithValue(error.message);
    }
  }
);

// 검색 결과 API 호출 thunk
export const fetchSearchResult = createAsyncThunk(
  'flows/fetchSearchResult',
  async (flowId, { rejectWithValue }) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/search/result/?flow_id=${flowId}`
      );
      if (!response.ok) throw new Error('Failed to fetch search result');
      const data = await response.json();
      // 반환된 data에서 search_result 배열 추출
      return { flowId, data: data.search_result };
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

//
// -------------------- Redux Slice --------------------
//

const initialState = {
  flows: {}, // Flow별 상태 관리 (flowId 기반 저장)
  searchResult: {}, // ← 여기에 검색 결과 저장 (flowId별)
  surrogateFeatureImportance: {},
  surrogateMatric: {},
  surrogateResult: {},
  priorities: {},
  optimizationData: {},
  histograms: {},
  properties: {}, // ✅ concat_csv_id를 키로 하는 properties 저장 공간 추가
  newCategories: {},
  status: 'idle',
  error: null,
};

const flowSlice = createSlice({
  name: 'flows',
  initialState,
  reducers: {
    initializeFlow: (state, action) => {
      const { flowId, concatCsvId = 3 } = action.payload; // ✅ 기본값 설정
      if (!state.flows[flowId]) {
        state.flows[flowId] = {
          datasets: [],
          concat_csv_id: concatCsvId, // ✅ 기본값 적용
        };
      }
    },
    updateFlow: (state, action) => {
      const { flowId, progress } = action.payload;
      if (state.flows[flowId]) {
        state.flows[flowId].progress = progress;
      }
    },
    setCurrentStep: (state, action) => {
      const { flowId, step } = action.payload;
      if (state.flows[flowId]) {
        state.flows[flowId].currentStep = step;
      }
    },
    setConcatCsvId: (state, action) => {
      const { flowId, concatCsvId } = action.payload;
      if (state.flows[flowId]) {
        state.flows[flowId].concat_csv_id = concatCsvId;
      }
    },
    updatePropertyCategory: (state, action) => {
      // action.payload: { flowId, property, newCategory }
      const { flowId, property, newCategory } = action.payload;
      // 먼저, 해당 flowId의 properties를 가져오고, 각 카테고리에서 property를 제거
      const propState = state.properties[flowId];
      if (propState) {
        // 삭제: 모든 카테고리에서 해당 property 제거
        Object.keys(propState).forEach(cat => {
          propState[cat] = propState[cat].filter(p => p !== property);
        });
        // 추가: newCategory에 추가 (만약 newCategory가 유효하다면)
        if (propState[newCategory]) {
          propState[newCategory].push(property);
        }
      }
    },
    initializeCategories: (state, action) => {
      const { flowId, properties } = action.payload;

      if (!state.properties[flowId]) {
        state.properties[flowId] = properties; // ✅ 기존 properties 저장
      }

      if (!state.newCategories[flowId]) {
        state.newCategories[flowId] = {}; // ✅ 새로운 category 저장 공간 초기화
      }
    },
    removeCategory: (state, action) => {
      const { flowId, property } = action.payload;
      // 해당 flowId 아래 newCategories에서 해당 property를 제거
      if (state.newCategories[flowId]) {
        delete state.newCategories[flowId][property];
      }
    },
    updateCategory: (state, action) => {
      const { flowId, property, category } = action.payload;

      if (!state.newCategories[flowId]) {
        state.newCategories[flowId] = {};
      }

      // ✅ property에 대한 새로운 category 정보 추가
      state.newCategories[flowId][property] = category;
    },
    updateOptimizationData: (state, action) => {
      const { flowId, property, newData, type } = action.payload;
      if (!state.optimizationData[flowId]) {
        state.optimizationData[flowId] = {};
      }
      state.optimizationData[flowId][property] = {
        ...state.optimizationData[flowId][property],
        ...newData,
        type,
      };
    },
    updatePriorities: (state, action) => {
      const { flowId, priorities } = action.payload;
      state.priorities[flowId] = priorities;
    },
    initializePriorities: (state, action) => {
      const { flowId } = action.payload;

      if (!state.priorities[flowId] && state.optimizationData[flowId]) {
        state.priorities[flowId] = Object.keys(state.optimizationData[flowId]);
      }
    },
  },
  extraReducers: builder => {
    builder
      .addCase(fetchFlowsByProject.fulfilled, (state, action) => {
        const { flows } = action.payload;
        state.flows = { ...state.flows, ...flows };
      })
      .addCase(fetchFlowsByProject.rejected, (state, action) => {
        console.error('❌ Failed to fetch flows in project', action.payload);
        state.error = action.payload;
      })
      .addCase(addFlowAsync.fulfilled, (state, action) => {
        const { flowId, flow } = action.payload;
        state.flows[flowId] = flow;
      })
      .addCase(deleteFlowAsync.fulfilled, (state, action) => {
        delete state.flows[action.payload];
      })
      .addCase(editFlowAsync.fulfilled, (state, action) => {
        const { flowId, flowName } = action.payload;
        if (state.flows[flowId]) {
          state.flows[flowId].flow_name = flowName;
        }
      })
      // ✅ Flow에 추가된 CSV 목록 조회 성공
      .addCase(fetchFlowDatasets.fulfilled, (state, action) => {
        const { flowId, datasets } = action.payload;
        if (!state.flows[flowId]) {
          state.flows[flowId] = { csv: [] };
        }
        state.flows[flowId].csv = datasets.map(dataset => dataset?.csvId);
      })
      .addCase(fetchFlowDatasets.rejected, (state, action) => {
        console.error('❌ Failed to fetch flow datasets:', action.payload);
        state.error = action.payload;
      })

      // ✅ Flow에 CSV 추가 성공
      .addCase(addCsvToFlow.fulfilled, (state, action) => {
        const { flowId, csvIds } = action.payload;

        if (!state.flows[flowId]) {
          state.flows[flowId] = { datasets: [] };
        }

        // 기존 csvId 리스트 가져오기 (기본적으로 빈 배열)
        const existingIds = state.flows[flowId].datasets || [];

        // 중복되지 않은 새로운 csvId만 추가
        const newDatasets = csvIds.filter(id => !existingIds.includes(id));

        // 최종적으로 csvId 배열을 유지
        state.flows[flowId].datasets = [...existingIds, ...newDatasets];
      })
      .addCase(addCsvToFlow.rejected, (state, action) => {
        console.error('❌ Failed to add CSV to flow:', action.payload);
        state.error = action.payload;
      })
      .addCase(fetchFlowProperties.fulfilled, (state, action) => {
        const { flowId, data } = action.payload;
        // API에서 받아온 dataset properties
        state.properties[flowId] = {
          numerical: data.numerical,
          categorical: data.categorical,
          text: data.text,
          unavailable: data.unavailable,
        };
        // API에서 받아온 새로운 카테고리 정보를 newCategories에 저장
        const categories = {};
        data.environmental.forEach(
          prop => (categories[prop] = 'environmental')
        );
        data.controllable.forEach(prop => (categories[prop] = 'controllable'));
        data.output.forEach(prop => (categories[prop] = 'output'));
        state.newCategories[flowId] = categories;
      })
      .addCase(fetchPropertyTypes.fulfilled, (state, action) => {
        const { flowId, data } = action.payload;
        // API에서 받아온 dataset properties
        state.properties[flowId] = {
          numerical: data.numerical,
          categorical: data.categorical,
          text: data.text,
          unavailable: data.unavailable,
        };
      })
      .addCase(fetchFlowProperties.rejected, (state, action) => {
        console.error('Flow properties 받아오기 실패:', action.payload);
      })
      .addCase(fetchFlowHistograms.fulfilled, (state, action) => {
        const { flowId, histograms } = action.payload;
        state.histograms[flowId] = histograms;
      })
      .addCase(fetchFlowHistograms.rejected, (state, action) => {
        console.error('❌ Failed to fetch histograms:', action.payload);
        state.error = action.payload;
      })
      .addCase(fetchPropertyHistograms.fulfilled, (state, action) => {
        const { flowId, column_name, histograms } = action.payload;
        if (!state.histograms) {
          state.histograms = {};
        }
        if (!state.histograms[flowId]) {
          state.histograms[flowId] = {};
        }
        state.histograms[flowId][column_name] = histograms;
      })
      .addCase(fetchPropertyHistograms.rejected, (state, action) => {
        console.error('fetchPropertyHistograms rejected:', action.payload);
        // 필요한 에러 처리를 추가
      })
      .addCase(savePropertyCategories.fulfilled, (state, action) => {
        console.log(
          '✅ Property categories successfully updated:',
          action.payload
        );
      })
      .addCase(savePropertyCategories.rejected, (state, action) => {
        console.error(
          '❌ Failed to update property categories:',
          action.payload
        );
        state.error = action.payload;
      })
      .addCase(fetchOptimizationData.fulfilled, (state, action) => {
        const { flowId, property, type, data } = action.payload;
        if (!state.optimizationData[flowId]) {
          state.optimizationData[flowId] = {};
        }

        const defaultGoal =
          type === 'controllable' ? 'No Optimization' : 'Fit to Property';

        // 매핑 객체 정의
        const goalMapping = {
          1: 'No Optimization',
          2: 'Maximize',
          3: 'Minimize',
          4: 'Fit to Range',
          5: 'Fit to Property',
        };

        // data.goal 값이 숫자이면 매핑 객체로, 아니면 문자열이면 그대로 사용, 없으면 defaultGoal
        const goalStr =
          typeof data.optimize_goal === 'number'
            ? goalMapping[data.optimize_goal] || defaultGoal
            : typeof data.optimize_goal === 'string'
            ? data.optimize_goal
            : defaultGoal;

        state.optimizationData[flowId][property] = {
          minimum_value:
            data.minimum_value !== undefined ? data.minimum_value : '',
          maximum_value:
            data.maximum_value !== undefined ? data.maximum_value : '',
          goal: goalStr,
          type: type,
          order: data.optimize_order,
        };
      })
      .addCase(postOptimizationData.fulfilled, (state, action) => {
        // action.payload는 { flowId, property, type, payload, data } 형태로 반환됨
        const { flowId, property, data } = action.payload;
        // 만약 해당 flowId가 아직 없으면 초기화
        if (!state.optimizationData[flowId]) {
          state.optimizationData[flowId] = {};
        }
        // 해당 property의 데이터를 업데이트 (서버에서 반환한 data를 사용)
        state.optimizationData[flowId][property] = data;
      })
      .addCase(postOptimizationData.rejected, (state, action) => {
        console.error('POST Optimization Data 실패:', action.payload);
      })
      .addCase(postOptimizationOrder.fulfilled, (state, action) => {
        // 성공 시 추가 처리 (필요 시)
      })
      .addCase(postOptimizationOrder.rejected, (state, action) => {
        // 에러 처리 (필요 시)
      })
      .addCase(fetchSurrogateFeatureImportance.fulfilled, (state, action) => {
        const { flowId, data } = action.payload;
        state.surrogateFeatureImportance[flowId] = data;
      })
      .addCase(fetchSurrogateFeatureImportance.rejected, (state, action) => {
        state.error = action.payload;
      })
      .addCase(fetchSurrogateMatric.fulfilled, (state, action) => {
        const { flowId, data } = action.payload;
        state.surrogateMatric[flowId] = data;
      })
      .addCase(fetchSurrogateMatric.rejected, (state, action) => {
        state.error = action.payload;
      })
      .addCase(fetchSurrogateResult.fulfilled, (state, action) => {
        const { flowId, data } = action.payload;
        state.surrogateResult[flowId] = data;
      })
      .addCase(fetchSurrogateResult.rejected, (state, action) => {
        state.error = action.payload;
      })
      .addCase(fetchSearchResult.fulfilled, (state, action) => {
        const { flowId, data } = action.payload;
        // flowId별로 검색 결과 저장
        state.searchResult[flowId] = data;
        // 필요에 따라 loading 플래그를 false로 전환 (만약 따로 관리 중이면)
      })
      .addCase(fetchSearchResult.rejected, (state, action) => {
        state.error = action.payload || action.error.message;
        // 에러 처리 및 loading 상태 false 전환
      });
  },
});

export const {
  initializeFlow,
  updateFlow,
  setCurrentStep,
  initializeCategories,
  updatePropertyCategory,
  removeCategory,
  updateCategory,
  updateOptimizationData,
  updatePriorities,
  initializePriorities,
} = flowSlice.actions;
export default flowSlice.reducer;
