import { configureStore } from '@reduxjs/toolkit';
import projectsReducer from './features/projectSlice';
import flowReducer from './features/flowSlice'; // flowSlice 추가

const store = configureStore({
  reducer: {
    projects: projectsReducer,
    flows: flowReducer, // flowSlice 통합
  },
  middleware: getDefaultMiddleware => getDefaultMiddleware(), // thunk는 이미 Redux Toolkit 기본 내장
});

export default store;
