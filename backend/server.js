import express from 'express';
import connectDB from './config/db.js';

connectDB();

const app = express();

app.get('/', (req, res) => {
  res.send('API is running...');
});

