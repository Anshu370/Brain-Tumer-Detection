import mongoose from "mongoose";
import { DB_NAME } from "../constant.js";

const connectDb=async()=>{
    try{
        const c=await mongoose.connect(`${process.env.MONGODB_URI}/${DB_NAME}`);
        console.log(`\n MongoDB connected ${c.connection.host}`);
    } catch(error){
        console.log("Error ",error);
        process.exit(1)
    }
}

export default connectDb;