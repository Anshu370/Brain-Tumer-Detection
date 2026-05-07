import mongoose from 'mongoose';

const appointmentSchema = new mongoose.Schema({
  patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  doctorId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  date: { type: Date, required: true },
  time: { type: String, required: true },
  note: {type: String},
  status:
        {
            type: String,
            enum: ["Approved","Pending","Rejected"],
            default: "Pending"
        }
}, { timestamps: true });

export const Appointment = mongoose.model('Appointment', appointmentSchema);
