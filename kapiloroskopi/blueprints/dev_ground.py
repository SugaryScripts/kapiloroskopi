from flask import Blueprint, render_template, current_app, flash, session, redirect, url_for, send_from_directory, \
  jsonify, request

blueprint = Blueprint("devs_field", __name__)