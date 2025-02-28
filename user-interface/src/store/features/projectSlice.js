import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;

//
// -------------------- Projects API Thunks --------------------
//

// 프로젝트 목록 가져오기
export const fetchProjects = createAsyncThunk(
  'projects/fetchProjects',
  async (_, { rejectWithValue }) => {
    try {
      const response = await fetch(`${API_BASE_URL}/projects/`);
      if (!response.ok) throw new Error('Failed to fetch projects');

      const projects = await response.json();

      // ✅ Redux에서 projectId 형태로 변환
      const formattedProjects = projects.projects.map(({ id, ...rest }) => ({
        projectId: id, // ✅ id → projectId
        ...rest,
      }));

      return formattedProjects;
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

// 프로젝트 추가
export const addProjectAsync = createAsyncThunk(
  'projects/addProject',
  async ({ name, description }, { rejectWithValue }) => {
    try {
      const response = await fetch(`${API_BASE_URL}/projects/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description }),
      });
      if (!response.ok) throw new Error('Failed to add project');

      const newProject = await response.json();

      return {
        projectId: newProject.project_id, // ✅ Redux에서 projectId로 저장
        name: name,
        description: description,
      };
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

// 프로젝트 수정
export const editProjectAsync = createAsyncThunk(
  'projects/editProject',
  async ({ projectId, name, description }, { rejectWithValue }) => {
    try {
      const response = await fetch(`${API_BASE_URL}/projects/`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_id: projectId, name, description }), // ✅ 백엔드 요청 시 project_id 사용
      });
      if (!response.ok) throw new Error('Failed to edit project');

      return { projectId, name, description }; // ✅ Redux에 저장 시 projectId 유지
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

// 프로젝트 삭제
export const deleteProjectAsync = createAsyncThunk(
  'projects/deleteProject',
  async ({ projectId }, { rejectWithValue }) => {
    try {
      const response = await fetch(`${API_BASE_URL}/projects/`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_id: projectId }), // ✅ 백엔드 요청 시 project_id 사용
      });
      if (!response.ok) throw new Error('Failed to delete project');

      return { projectId }; // ✅ Redux에 저장 시 projectId 유지
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

//
// -------------------- CSV (Dataset) API Thunks --------------------
//

// ✅ 프로젝트별 CSV 파일 조회
export const fetchCsvFilesByProject = createAsyncThunk(
  'datasets/fetchCsvFilesByProject',
  async (projectId, { rejectWithValue }) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/csvs/?project_id=${projectId}`
      );
      if (!response.ok) throw new Error('Failed to fetch CSV files');

      const datasets = await response.json(); // ✅ API 응답 객체

      if (!datasets.csvs || !Array.isArray(datasets.csvs)) {
        throw new Error('Invalid data format: Expected csvs array');
      }

      // ✅ csv_id → csvId 변환 후 Redux 상태에 저장
      const formattedDatasets = datasets.csvs.map(
        ({ id, project, ...rest }) => ({
          csvId: id, // ✅ id → csvId
          projectId: project, // ✅ project → projectId
          ...rest,
        })
      );

      return { projectId, datasets: formattedDatasets };
    } catch (error) {
      console.error('❌ CSV 파일 조회 실패:', error.message);
      return rejectWithValue(error.message);
    }
  }
);

// ✅ CSV 파일 업로드
export const uploadCsvFile = createAsyncThunk(
  'datasets/uploadCsvFile',
  async ({ projectId, file, writer }, { rejectWithValue }) => {
    try {
      const formData = new FormData();
      formData.append('csv_file', file);
      formData.append('writer', writer);
      formData.append('project_id', projectId);

      const response = await fetch(`${API_BASE_URL}/csvs/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Failed to upload CSV file');

      const newCsv = await response.json();

      return {
        projectId,
        csvId: newCsv.csv_id, // ✅ Redux에서 csvId 형태로 저장
        ...newCsv,
      };
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

// ✅ CSV 파일 삭제
export const deleteCsvFile = createAsyncThunk(
  'datasets/deleteCsvFile',
  async ({ projectId, csvId }, { rejectWithValue }) => {
    try {
      const response = await fetch(`${API_BASE_URL}/csvs/`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_id: csvId }), // ✅ 요청 Body에 file_id 포함
      });

      if (!response.ok) throw new Error('Failed to delete CSV file');

      return { projectId, csvId }; // ✅ Redux 상태에서 projectId 기반으로 삭제 처리
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

const projectSlice = createSlice({
  name: 'projects',
  initialState: {
    projects: [], // ✅ projectId 형태로 저장
    datasets: {}, // ✅ csvId 형태로 저장
    status: 'idle',
    error: null,
  },
  reducers: {},
  extraReducers: builder => {
    builder
      .addCase(fetchProjects.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.projects = action.payload; // ✅ projectId로 변환된 상태 저장
      })
      .addCase(addProjectAsync.fulfilled, (state, action) => {
        state.projects.push(action.payload);
      })
      .addCase(editProjectAsync.fulfilled, (state, action) => {
        const { projectId, name, description } = action.payload;
        const project = state.projects.find(
          proj => proj.projectId === projectId
        );
        if (project) {
          project.name = name;
          project.description = description;
        }
      })
      .addCase(deleteProjectAsync.fulfilled, (state, action) => {
        state.projects = state.projects.filter(
          proj => proj.projectId !== action.payload.projectId
        );
      })

      // ✅ 프로젝트별 CSV 파일 가져오기
      .addCase(fetchCsvFilesByProject.fulfilled, (state, action) => {
        const { projectId, datasets } = action.payload;
        state.datasets[projectId] = datasets; // ✅ csvId로 변환된 데이터 저장
      })
      .addCase(fetchCsvFilesByProject.rejected, (state, action) => {
        state.error = action.payload;
      })

      // ✅ CSV 파일 업로드 성공 시 Redux 상태 업데이트
      .addCase(uploadCsvFile.fulfilled, (state, action) => {
        const { projectId, csvId } = action.payload;
        if (!state.datasets[projectId]) {
          state.datasets[projectId] = [];
        }
        state.datasets[projectId].push(action.payload); // ✅ csvId로 저장
      })
      .addCase(uploadCsvFile.rejected, (state, action) => {
        state.error = action.payload;
      })

      // ✅ CSV 파일 삭제 성공 시 Redux 상태 업데이트
      .addCase(deleteCsvFile.fulfilled, (state, action) => {
        const { projectId, csvId } = action.payload;

        if (state.datasets[projectId]) {
          state.datasets[projectId] = state.datasets[projectId].filter(
            dataset => dataset.csvId !== csvId
          );
        }
      })
      .addCase(deleteCsvFile.rejected, (state, action) => {
        state.error = action.payload;
      });
  },
});

export default projectSlice.reducer;
