import mongoose from 'mongoose';

const reportSchema = new mongoose.Schema({
  patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  wordict: { type: String, required: true },
  date: { type: Date, default: Date.now }
}, { timestamps: true });

export const Report = mongoose.model('Report', reportSchema);