import axios from 'axios';

const API_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
});

export const uploadResume = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/extract_text', formData);
  return response.data;
};

export const scoreResume = async (file, jobDescription) => {
  const formData = new FormData();
  formData.append('resume', file);
  formData.append('job_description', jobDescription);
  const response = await api.post('/score_resume', formData);
  return response.data;
};

export const parseResume = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/parse_resume', formData);
  return response.data;
};

export default api;
