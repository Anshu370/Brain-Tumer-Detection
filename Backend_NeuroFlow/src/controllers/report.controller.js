import {asyncHandler} from "../utils/asyncHandler.js";
import {ApiError} from "../utils/ApiError.js"
import { Report } from "../models/report.model.js";
import { ApiResponse } from "../utils/ApiResponse.js";
import axios from 'axios'

const createReport = asyncHandler(async (req, res) => {
    const userId = req.user._id;
    console.log(req);
    if (!req.file) {
        throw new ApiError(400, "Image is required");
    }

    // If using Cloudinary/S3
    const imageUrl = req.file.path;

    // 👉 Call your hosted ML API
    const mlResponse = await axios.post(
        "https://anshu370-brain-tumer-detection.hf.space/predict",
        {
            imageUrl   // OR send file if API expects file
        },
        {
            headers: {
                Authorization: `Bearer ${process.env.ML_API_KEY}`
            }
        }
    );

    console.log(mlResponse);

    

    // // 👉 Save report
    // const report = await Report.create({
    //     patientId: userId,
    //     imageUrl,
    //     tumorType,
    //     confidence,
    //     recommendedDoctors: doctors.map(d => d._id)
    // });

    // const populatedReport = await Report.findById(report._id)
    //     .populate("recommendedDoctors", "name email specialization");

    return res.status(201).json(
        new ApiResponse(201, mlResponse, "Report generated successfully")
    );
});

const getReportsByUserId = asyncHandler(async (req, res) => {
    const userId = req.user._id;

    if (!userId) {
        throw new ApiError(401, "Unauthorized request");
    }

    const reports = await Report.find({ patientId: userId })
        .populate("patientId", "name email")
        .sort({ createdAt: -1 }); // latest first

    return res.status(200).json(
        new ApiResponse(200, reports, "Reports fetched successfully")
    );
});

export {getReportsByUserId,createReport};
