import {asyncHandler} from "../utils/asyncHandler.js";
import {ApiError} from "../utils/ApiError.js"
import { ApiResponse } from "../utils/ApiResponse.js";
import {User} from "../models/user.model.js"
import { Appointment } from "../models/appointment.model.js";

const bookAppointment = asyncHandler(async (req, res) => {
    const { doctorId, date, time, note } = req.body;
    const patientId = req.user._id;

    // Validate required fields
    if (!doctorId || !date || !time) {
        throw new ApiError(400, "doctorId, date and time are required");
    }

    // Prevent booking yourself (edge case)
    if (doctorId.toString() === patientId.toString()) {
        throw new ApiError(400, "You cannot book an appointment with yourself");
    }

    // Check doctor exists
    const doctor = await User.findById(doctorId);
    if (!doctor || doctor.role !== "doctor") {
        throw new ApiError(404, "Doctor not found");
    }

    // Normalize date (remove time part)
    const appointmentDate = new Date(date);
    appointmentDate.setHours(0, 0, 0, 0);

    // Combine date + time safely
    const appointmentDateTime = new Date(`${date}T${time}`);

    // Prevent past booking
    if (appointmentDateTime < new Date()) {
        throw new ApiError(400, "Cannot book appointment in the past");
    }

    // Check slot conflict (only active bookings)
    const existingAppointment = await Appointment.findOne({
        doctorId,
        date: appointmentDate,
        time,
        status: { $in: ["Pending", "Approved"] }
    });

    if (existingAppointment) {
        throw new ApiError(409, "This time slot is already booked");
    }

    // Create appointment
    const appointment = await Appointment.create({
        doctorId,
        patientId,
        date: appointmentDate,
        time,
        note,
    });

    // Populate response
    const bookedAppointment = await Appointment.findById(appointment._id)
        .populate("doctorId", "name email specialization place")
        .populate("patientId", "name email");

    return res.status(201).json(
        new ApiResponse(
            201,
            bookedAppointment,
            "Appointment booked successfully"
        )
    );
});

const getUserAppointments = asyncHandler(async (req, res) => {
    const userId = req.user._id;

    if (!userId) {
        throw new ApiError(401, "Unauthorized request");
    }

    // Fetch appointments where user is either patient or doctor
    const appointments = await Appointment.find({
        $or: [
            { patientId: userId },
            { doctorId: userId }
        ]
    })
    .populate("patientId", "name email")
    .populate("doctorId", "name email")
    .sort({ date: 1, time: 1 }); // upcoming first

    return res.status(200).json(
        new ApiResponse(
            200,
            appointments,
            "Appointments fetched successfully"
        )
    );
});

const updateAppointmentStatus = asyncHandler(async (req, res) => {
    const { appointmentId } = req.params;
    const { status } = req.body;

    const userId = req.user._id;

    // Validate status
    const validStatuses = ["Approved", "Rejected"];
    if (!validStatuses.includes(status)) {
        throw new ApiError(400, "Invalid status");
    }

    // Find appointment
    const appointment = await Appointment.findById(appointmentId);

    if (!appointment) {
        throw new ApiError(404, "Appointment not found");
    }

    // Ensure only assigned doctor can update
    if (appointment.doctorId.toString() !== userId.toString()) {
        throw new ApiError(403, "Not authorized to update this appointment");
    }

    // Prevent re-updating
    if (appointment.status !== "Pending") {
        throw new ApiError(400, "Appointment already updated");
    }

    // Update status
    appointment.status = status;
    await appointment.save();

    return res.status(200).json(
        new ApiResponse(200, appointment, `Appointment ${status}`)
    );
});

export {bookAppointment,getUserAppointments,updateAppointmentStatus};
