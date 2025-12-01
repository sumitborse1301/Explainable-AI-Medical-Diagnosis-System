
import streamlit as st
from datetime import datetime
import json
import os
import io
from PIL import Image
import base64
from prompts import ANALYSIS_PROMPT, SYSTEM_MESSAGE, LITERATURE_SYSTEM_MESSAGE, FALLBACK_RESPONSE, ERROR_RESPONSE, ERROR_REFERENCES


from util_simple import(
    process_file,
    analyze_image,
    generate_heatmap,
    save_analysis,
    get_latest_analyses,
    generate_report,
    search_pubmed,
    generate_statistics_report
)

from chat_system import render_chat_interface, create_manual_chat_room

from report_qa_chat import ReportQASystem, ReportQAChat
from qa_interface import render_qa_chat_interface

st.set_page_config(
    page_title="Medical Image Analysis Platform",
    page_icon="🏥",
    layout="wide"
) 

if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""
if "file_data" not in st.session_state:
    st.session_state.file_data = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "file_type" not in st.session_state:
    st.session_state.file_type = None
if "OPENAI_AI_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = None

# cached list for the sidebar Recent Analyses (keeps UI responsive)
if "latest_analyses" not in st.session_state:
    st.session_state.latest_analyses = get_latest_analyses(limit=5)


st.title("🏥 Explainable AI Assisted Doctor Diagnosis System")
st.markdown("Upload medical images for AI-Powered analysis and collaborate with colleagues")


with st.sidebar:
    st.header("Configuration")

    # API Key Input
    api_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.openai_key,
        type="password",
        help="Enter your OpenAI API key to enable image analysis" 
    )

    if api_key:
        st.session_state.openai_key = api_key
        st.session_state.OPENAI_API_KEY = api_key
    
    # Explainable AI Options
    st.subheader("Analysis Options")
    enable_xai = st.checkbox("Enable Explainable AI", value=True)
    include_references = st.checkbox("Include Medical References", value=True)

    # Recent analyses
    st.subheader("Recent Analyses")

    # Prefer session state cached list (updated right after saving)
    recent_analyses = st.session_state.get("latest_analyses")
    if recent_analyses is None:
        recent_analyses = get_latest_analyses(limit=5)

    # Display nicely formatted date
    from datetime import datetime
    for analysis in recent_analyses[:5]:
        d = analysis.get('date', '') or ''
        date_str = d[:10]
        try:
            # If ISO datetime is present, format cleanly
            date_str = datetime.fromisoformat(d).strftime("%Y-%m-%d")
        except Exception:
            date_str = d[:10] or "Unknown date"
        st.caption(f"{analysis.get('filename', 'Unknown')} - {date_str}")




    # Statistics report
    if st.button("Generate Statistics Report"):
        stats_report = generate_statistics_report()
        if stats_report:
            # Create download link
            b64_pdf = base64.b64encode(stats_report.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="statistics_report.pdf">Download Statistics Report</a>'
            st.markdown(href, unsafe_allow_html=True)


tab1, tab2, tab3, tab4 = st.tabs(["Image Upload & Analysis", "Collaboration", "Report Q&A", "Reports"])    

with tab1:
    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload a medical image (JPEG, PNG, DICOM, NIfTI)",
        type=["jpg", "jpeg", "png", "dcm", "nii", "nii.gz"]
    )


    if uploaded_file:
        # Process the file
        try:
            file_data = process_file(uploaded_file)

            if file_data:
                st.session_state.file_data = file_data
                st.session_state.file_name = uploaded_file.name
                st.session_state.file_type = file_data["type"]

                # Display the image
                st.image(file_data["data"], caption=f"Uploaded {file_data['type']} ", use_container_width=True)

                # Analysis button
                if st.button("Analyze Image") and st.session_state.openai_key:
                    with st.spinner("Analyzing Image...."):

                        # Run image analysis
                        analysis_results = analyze_image(
                            file_data["data"],
                            st.session_state.openai_key,
                            enable_xai=enable_xai
                        )

                        # Store the analysis (this writes to analysis_store.json)
                        analysis_results = save_analysis(analysis_results, filename=uploaded_file.name)

                        # Update session state so UI will reflect new analysis immediately
                        st.session_state.analysis_results = analysis_results

                        # Update cached recent list so sidebar reads the new data immediately
                        st.session_state.latest_analyses = get_latest_analyses(limit=5)

                        # Show small confirmation to user
                        st.success("Analysis saved and added to Recent Analyses ✅")







                        
                        # Display results
                        st.subheader("Analysis Results")
                        st.markdown(analysis_results["analysis"])

                        # Show findings if available
                        if analysis_results.get("findings"):
                            st.subheader("Key Findings")
                            for idx, finding in enumerate(analysis_results["findings"], 1):
                                st.markdown(f"{idx}. {finding}")
                        
                        # Show keywords if available
                        if analysis_results.get("keywords"):
                            st.subheader("Keywords")
                            st.markdown(f"*{', '.join(analysis_results['keywords'])}*")

                        # Generate heatmap if XAI is enabled
                        if enable_xai:
                            st.subheader("Explainable AI Visualization")
                            overlay, generate_heatmap = generate_heatmap(file_data["array"])
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.image(overlay, caption="Heatmap Overlay", use_container_width=True)
                            
                            with col2:
                                st.image(overlay, caption="Raw Heatmap", use_container_width=True)

                        # Show medical references if enabled
                        if include_references and analysis_results.get("keywords"):
                            st.subheader("Relevant Medical Literature")
                            references = search_pubmed(analysis_results["keywords"], max_results=3)

                            for ref in references:
                                st.markdown(f"- **{ref['title']}** \n{ref['journal']}, {ref['year']} (PMID: {ref['id']})")

                        # Generate PDF Report
                        st.subheader("Report Generation")
                        pdf_buffer = generate_report(analysis_results, include_references=include_references)

                        # Create download link for the PDF
                        b64_pdf = base64.b64encode(pdf_buffer.read()).decode()
                        href = f'<a href="data:application/pdf;base64, {b64_pdf}" download="medical_report_{datetime.now().strftime("%Y%m%d")}.pdf">Download PDF Report</a>'
                        st.markdown(href, unsafe_allow_html=True)

                        # Option to start a discussion
                        st.subheader("Collaborate")
                        col1, col2 = st.columns(2)

                        with col1:
                            if st.button("Start Case Discussion"):
                                # Create a chat room with the default name
                                case_description = f"{uploaded_file.name} analysis"
                                if "findings" in analysis_results and analysis_results["findings"]:
                                    case_description = analysis_results
                                    ["findings"] [0]

                                # Use the file type (which we know exits now)
                                case_id = f"{file_data['type'].upper()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                                created_case_id = create_manual_chat_room
                                ("Dr. Anonymous", case_description)
                                st.session_state.current_case_id = created_case_id
                                st.rerun()

                        with col2:
                            if st.button("Start Q&A Session"):
                                # Create a Q&A room for this analysis
                                if "qa_chat" not in st.session_state:
                                    st.session_state.qa_chat = ReportQAChat()

                                room_name = f"Q&A for {uploaded_file.name}"
                                created_qa_id = st.session_state.qa_chat.create_qa_room("Dr. Anonymous", room_name)
                                st.session_state.current_qa_id = created_case_id
                                st.rerun

                elif not st.session_state.openai_key:
                    st.warning("Please enter your OpenAI API Key in the sidebar to enable analysis")

            else:
                st.error("Unable to process the uploaded file")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    elif "analysis_results" in st.session_state and st.session_state.analysis_results:
        st.subheader("Previous Analysis Results")
        st.markdown(st.session_state.analysis_results["analysis"]) 

        if "findings" in st.session_state.analysis_results:
            st.subheader("Key Findings")
            for idx, finding in enumerate(st.session_state.analysis_results["findings"], 1):
                st.markdown(f"{idx}. {finding}")


        # Generate PDF Report for previous analysis
        st.subheader("Report")
        if st.button("Generate PDF Report"):
            pdf_buffer = generate_report(st.session_state.analysis_results, include_references=include_references)


            # Create download link for the PDF
            b64_pdf = base64.b64encode(pdf_buffer.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="medical_report_{datetime.now().strftime("%Y%m%d")}.pdf">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)


with tab2:
    # Render the chat interface for collaboration
    try:
        render_chat_interface()
    except Exception as e:
        st.error(f"Error in chat interface {str(e)}")
        st.info("If you're trying to create a new discussion, please upload and analyze an image first.")

        # Offer a direct way to create a discussion without an image
        st.subheader("Create Discussion Without Image")
        manual_creator = st.text_input("Your Name", value="Dr. Anonymous")
        manual_description = st.text_input("Case Description")

        if st.button("Create Manual Discussion") and manual_description:
            case_id = create_manual_chat_room(manual_creator, manual_description)
            st.session_state.current_case_id = case_id
            st.rerun()


with tab3:
    # Render the QA Chat interface
    render_qa_chat_interface()

with tab4:
    # Report and Analytics section
    st.subheader("Medical Reports & Analytics")

    # Analysis history
    st.markdown("### Analysis History")
    recent_analyses = get_latest_analyses(limit=10)

    if recent_analyses:
        for idx, analysis in enumerate(recent_analyses, 1):
            with st.expander(f"{idx}. {analysis.get('filename', 'Unkown')} - {analysis.get('date', '') [:10]}"):
                st.markdown(analysis.get("analysis", "No analysis available"))

                if analysis.get("findings"):
                    st.markdown("**Key Findings:**")
                    for finding_idx, finding in enumerate(analysis["findings"], 1):
                        st.markdown(f"{finding_idx}. {finding}")
                col1, col2 = st.columns(2)

                # Generate individual report
                with col1:
                    if st.button(f"Generate Report #{idx}"):
                        pdf_buffer = generate_report(analysis, include_references=include_references)

                        # Create download link
                        b64_pdf = base64.b64encode(pdf_buffer.read()).decode()
                        report_name = f"report_{analysis.get('id', 'unknown')[:8]}.pdf" 

                """with col2:
                    if st.button(f"Q&A on Report #{idx}"):
                        # Create a QA room specificaly for this report
                        if "qa_chat" not in st.session_state:
                            st.session_state.qa_chat = ReportQAChat()

                            report_name = f"Q&A for {analysis.get('filename', 'Unkonwn')}"
                            created_qa_id = st.session_state.qa_chat.create_qa_room
                            ("Dr. Anonymous", report_name)
                            st.session_state.current_qa_id = created_qa_id

                            # Switch to QA tab
                            st.rerun()"""
    
    else:
        st.info("No previous analyses found. Upload and analyze an image to get started.")

    # Statistics section
    st.markdown("### Statistics")

    # Generate statistics report 
    if st.button("Generate Comprehensive Statistics"):
        stats_report = generate_statistics_report()
        if stats_report:
            # Create download link
            b64_pdf = base64.b64encode(stats_report.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="statistics_report.pdf">Download Statistics Report</a>'
            st.markdown(href, unsafe_allow_html=True)













